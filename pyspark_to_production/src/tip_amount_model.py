from dataclasses import dataclass, fields

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark_to_production.src.log_config import get_logger
import argparse
from pathlib import Path

logger = get_logger(__name__)


@dataclass
class TipAmountModelConfig:
    history_start_month: str = "201703"
    history_end_month: str = "201811"
    first_evening_hour: str = "17"
    last_evening_hour: str = "23"
    n_first_daily_rides_to_keep: int = 3
    test_fraction: float = 0.2


feature_cols = [
    "passenger_count",
    "trip_distance",
    "rate_code",
    "payment_type",
    "fare_amount",
    "tolls_amount",
    "imp_surcharge",
    "month",
    "day_of_week",
    "day_of_month",
    "store_and_fwd_flag_is_N",
]


class TipAmountModel:
    def __init__(self, config: TipAmountModelConfig) -> None:
        self.config = config
        self.sdfs = {}
        self.model = None
        self.feature_cols = feature_cols
        self.log_inits()

    def log_inits(self) -> None:
        logger.info("Initialized with configs: %s", self.config)

        spark_app_name = self.spark.sparkContext.getConf().get("spark.app.name")
        logger.info("Spark app name: %s", spark_app_name)
    
    @property
    def spark(self):
        # takes the currently running spark session or creates a local one
        return (
            SparkSession.builder.master("local[*]")
            .appName("local_run")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )

    def run(self) -> None:
        self.extract()
        self.transform()
        self.train()
        self.validate()
        self.load()

    def extract(self) -> None:
        logger.info("Extracting datasets")
        dataset_names = [
            "taxi_trip_data",
            "taxi_zone_geo",
        ]
        for dataset_name in dataset_names:
            logger.info("  %s...", dataset_name)
            self.read_dataset(dataset_name)

    def read_dataset(self, dataset_name: str) -> None:
        path_here = Path(__file__).resolve()
        file_path = path_here.parent.parent / "data" / f"{dataset_name}.csv"
        self.sdfs[dataset_name] = self.spark.read.csv(
            str(file_path), header=True, inferSchema=True
        )

    def transform(self) -> None:
        logger.info("Preparing the data for training")
        self.sdfs["prepared_data"] = (
            self.sdfs["taxi_trip_data"]
            .transform(self.filter_data)
            .transform(self.add_features)
        )

    def filter_data(self, sdf: DataFrame) -> DataFrame:
        return (
            sdf.dropDuplicates()
            .transform(self.limit_history_to_a_range)
            .transform(self.keep_evening_rides_only)
            .transform(self.exclude_airports_by_location, "pickup_location_id")
            .transform(self.exclude_airports_by_location, "dropoff_location_id")
            .transform(self.keep_first_n_daily_rides_only)
        )

    def limit_history_to_a_range(self, sdf: DataFrame) -> DataFrame:
        pickup_month = F.date_format(F.col("pickup_datetime"), "yyyyMM")
        # fmt: off
        return (
            sdf
            .filter(pickup_month > self.config.history_start_month)
            .filter(pickup_month <= self.config.history_end_month)
        )
        # fmt: on

    def keep_evening_rides_only(self, sdf: DataFrame) -> DataFrame:
        dropoff_hour = F.date_format(F.col("dropoff_datetime"), "HH")
        # fmt: off
        return (
            sdf
            .filter(dropoff_hour >= self.config.first_evening_hour)
            .filter(dropoff_hour <= self.config.last_evening_hour)
        )
        # fmt: on

    def exclude_airports_by_location(
        self, sdf: DataFrame, location_id_col_name: str
    ) -> DataFrame:
        # fmt: off
        sdf_zone_geo_airport = (
            self.sdfs["taxi_zone_geo"]
            .filter(F.lower(F.col("zone_name")).like("%airport%"))
        )
        return (
            sdf
            .join(
                sdf_zone_geo_airport,
                on=[F.col(location_id_col_name) == F.col("zone_id")],
                how="leftanti"
            )
        )
        # fmt: on

    def keep_first_n_daily_rides_only(self, sdf: DataFrame) -> DataFrame:
        pickup_date = F.date_format(F.col("pickup_datetime"), "yyyyMMdd")
        # fmt: off
        window = (
            Window
            .partitionBy("pickup_location_id", pickup_date)
            .orderBy(F.asc("pickup_datetime"))
        )
        return (
            sdf
            .withColumn("ride_number", F.row_number().over(window))
            .filter(F.col("ride_number") <= self.config.n_first_daily_rides_to_keep)
            .drop("ride_number")
        )
        # fmt: on

    @staticmethod
    def add_features(sdf: DataFrame) -> DataFrame:
        date_metrics = {
            "month": F.month,
            "day_of_week": F.dayofweek,
            "day_of_month": F.dayofmonth,
        }
        # fmt: off
        return (
            sdf
            .withColumns({
                name: func(F.col("pickup_datetime"))
                for name, func in date_metrics.items()
            })
            .withColumn(
                "store_and_fwd_flag_is_N",
                F.when(F.col("store_and_fwd_flag") == "N", 1).otherwise(0)
            )
        )
        # fmt: on

    def train(self) -> None:
        self.train_test_split()
        logger.info("Training the model")
        self.train_model()

    def train_test_split(self) -> None:
        # fmt: off
        self.sdfs["training"], self.sdfs["test"] = (
            self.sdfs["prepared_data"]
            .randomSplit(weights=[1 - self.config.test_fraction, self.config.test_fraction], seed=42)
        )
        # fmt: on

    def train_model(self) -> None:
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")

        rf = RandomForestRegressor(
            labelCol="tip_amount",
            featuresCol="features",
            predictionCol="prediction",
            numTrees=10,
            maxDepth=4,
            featureSubsetStrategy="auto",
            seed=42,
            bootstrap=True,
        )

        pipeline = Pipeline(stages=[assembler, rf])

        self.model = pipeline.fit(self.sdfs["training"])

    def validate(self) -> None:
        logger.info("Start validation")
        self.check_features_importances()
        self.check_evaluation_metrics()

    def check_features_importances(self) -> None:
        logger.info("  Features importances")
        importances = zip(
            self.feature_cols, self.model.stages[-1].featureImportances, strict=False
        )
        importances_sorted = sorted(importances, key=lambda item: item[1], reverse=True)
        for name, importance in importances_sorted:
            logger.info("%22s = %.2g", name, importance)

    def check_evaluation_metrics(self) -> None:
        for set_name in ["training", "test"]:
            logger.info("  Evaluation on the %s set", set_name)
            self.evaluate_on_dataset(set_name)

    def evaluate_on_dataset(self, set_name: str) -> None:
        evaluator = RegressionEvaluator()
        evaluator.setPredictionCol("prediction")
        evaluator.setLabelCol("tip_amount")

        evaluation_metrics = ["rmse", "mae", "r2"]

        sdf_predictions = self.model.transform(self.sdfs[set_name])

        for metric_name in evaluation_metrics:
            value = evaluator.evaluate(
                sdf_predictions, {evaluator.metricName: metric_name}
            )
            logger.info("%8s = %.2g", metric_name, value)

    def load(self) -> None:
        logger.info("Saving the model")
        self.model.write().overwrite().save("../data/model")
        logger.info("The model is saved")


def dataclass_to_argparser(cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    for f in fields(cls):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            default=f.default,
            type=f.type,
        )

    return parser

if __name__ == "__main__":

    parser = dataclass_to_argparser(TipAmountModelConfig)
    args = parser.parse_args()
    config = TipAmountModelConfig(**vars(args))
    job = TipAmountModel(config)
    job.run()
