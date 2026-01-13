from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TypeVar

import pytest
from pyspark.sql import Row, SparkSession

from pyspark_to_production.src.tip_amount_model import (
    TipAmountModel,
    TipAmountModelConfig,
)

CT = TypeVar("CT")


@dataclass
class Trip:
    vendor_id: int = 1
    pickup_datetime: datetime = datetime(2018, 2, 4, 18, 0, 0, tzinfo=timezone.utc)
    dropoff_datetime: datetime = datetime(2018, 2, 4, 19, 30, 0, tzinfo=timezone.utc)
    passenger_count: int = 2
    trip_distance: float = 50.2
    rate_code: int = 3
    store_and_fwd_flag: str = "N"
    payment_type: int = 1
    fare_amount: float = 10.5
    extra: float = 0.1
    mta_tax: float = 0.5
    tip_amount: float = 0.8
    tolls_amount: float = 0.1
    imp_surcharge: float = 1.2
    total_amount: float = 15.2
    pickup_location_id: int = 1
    dropoff_location_id: int = 2


@dataclass
class ZoneGeo:
    zone_id: int = 1
    zone_name: str = "Snack Zone"
    borough: str = "Food Borough"


def generate_rows(
    data_class: type[CT],
    data: list[tuple] | None = None,
    columns: list[str] | None = None,
) -> list[Row]:
    data = data or [()]
    columns = columns or []

    generated_rows = []
    for record in data:
        record_dict = dict(zip(columns, record, strict=False))
        record_class = data_class(**record_dict)
        record_row = Row(**asdict(record_class))
        generated_rows.append(record_row)
    return generated_rows


def is_subset(a: list, b: list) -> bool:
    return set(a) <= set(b)


def test_add_features_column_names(spark: SparkSession) -> None:
    columns = ["pickup_datetime", "store_and_fwd_flag"]
    data = [
        (datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc), "Y"),
        (datetime(2021, 6, 15, 9, 30, 0, tzinfo=timezone.utc), "N"),
    ]

    tip_model = TipAmountModel(TipAmountModelConfig())

    tip_model.sdfs["taxi_trip_data"] = spark.createDataFrame(
        generate_rows(Trip, data, columns)
    )
    tip_model.sdfs["taxi_zone_geo"] = spark.createDataFrame(generate_rows(ZoneGeo))

    assert not is_subset(
        tip_model.feature_cols, tip_model.sdfs["taxi_trip_data"].columns
    )

    tip_model.transform()
    assert is_subset(tip_model.feature_cols, tip_model.sdfs["prepared_data"].columns)


@pytest.mark.parametrize(
    ("pickup_location_id", "dropoff_location_id", "n_expected_rows"),
    [
        (1, 1, 1),
        (100, 1, 0),
        (1, 100, 0),
        (100, 100, 0),
    ],
)
def test_exclude_airports_by_location(
    spark: SparkSession,
    pickup_location_id: int,
    dropoff_location_id: int,
    n_expected_rows: int,
) -> None:
    columns = ["pickup_location_id", "dropoff_location_id"]
    data = [(pickup_location_id, dropoff_location_id)]

    tip_model = TipAmountModel(TipAmountModelConfig())

    # no airports
    tip_model.sdfs["taxi_trip_data"] = spark.createDataFrame(
        generate_rows(Trip, data, columns)
    )
    tip_model.sdfs["taxi_zone_geo"] = spark.createDataFrame(
        generate_rows(ZoneGeo, [(100, "terrestrial")], ["zone_id", "zone_name"])
    )

    tip_model.transform()
    assert tip_model.sdfs["prepared_data"].count() == 1

    # all except one have airports
    tip_model.sdfs["taxi_zone_geo"] = tip_model.spark.createDataFrame(
        generate_rows(ZoneGeo, [(100, "is airport or so")], ["zone_id", "zone_name"])
    )

    tip_model.transform()
    assert tip_model.sdfs["prepared_data"].count() == n_expected_rows
