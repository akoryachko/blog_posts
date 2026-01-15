<!-- Review by Inês -->
# From PySpark Notebook to Production-Ready Code

## Motivation

The gap between a working notebook and maintainable production code is significant, especially in teams where people specialize in different parts of the project lifecycle.
Situations where data scientists throw a proof‑of‑concept notebook over the fence to engineers are common. Engineers then take the logic as‑is, wrap it into a scheduled job, and move on.
This approach often results in solutions that are hard to understand, modify, and maintain.
The problem is amplified in big‑data applications: data scientists may lack production‑grade coding experience, while engineers may not feel confident untangling long, monolithic PySpark queries.

## Purpose
This post focuses on practical techniques for bridging the gap between a free‑form PySpark notebook and modular, production‑ready code.
The goal is twofold:
- Help data scientists write code that is more readable and reusable.
- Give engineers confidence to reshape PySpark logic so that it aligns with standard software engineering practices.

## Who this post is for
- Data scientists whose PySpark notebooks eventually need to run in production.
- Data engineers inheriting notebook‑based logic.
- Teams struggling with unclear ownership between experimentation and implementation.

## Approach
We will walk through an example notebook that trains a model to predict taxi ride tip amounts.
The notebook works, but most of the logic lives in a single long PySpark query that might be difficult to understand.
Even the author will have issues untangling the logic after some time at another project.

We will progressively refactor this notebook, breaking the logic into functions, modules, and tests.
Each stage represents a different level of maintainability.
You can stop at any stage once your project’s requirements are satisfied. Projects have very different needs: from a one‑off analysis with no maintenance requirements to a business‑critical system with strict SLAs.
The stages below intentionally scale from minimal to rigorous.

## Example project
Assume the following (artificial but realistic) task.
We want to train a model that predicts the tip amount for New York City taxi rides based on trip information. The requirements are:
- Use NYC taxi data (pickup/dropoff timestamps, locations, fares, etc.).
- Keep only the first three evening rides per pickup location per day.
- Exclude trips to or from airports.
- Track feature importance and evaluation metrics for each run.
- Retrain the model regularly.

## Stage 0. Notebook solution
*Suitable only for one‑time analysis or proof‑of‑concept work.*

The full prototype notebook can be found [here](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/prototype.ipynb). Set up instructions for running the notebook locally are in the [`README.md`](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/README.md) of the same repo.

At this stage, correctness matters more than structure. The notebook produces the desired output and maintainability is not yet a concern.

## Stage 1. Refactored notebook
*Suitable for code that needs to be rerun occasionally.*

Once the notebook produces correct results for a single run, it is time to make it readable and easier to reason about.
The main goals are:
- Clear structure
- Reduced duplication
- Traceable execution

### Step 1.1. Put code in functions
#### Extracting dataset loading
We start by removing duplication in dataset loading:
```python
from pyspark.sql import DataFrame

def read_dataset(dataset_name: str) -> DataFrame:
    file_path = f"../data/{dataset_name}.csv"
    return spark.read.csv(file_path, header=True, inferSchema=True)

def extract() -> dict[str, DataFrame]:
    dataset_names = [
        "taxi_trip_data",
        "taxi_zone_geo",
    ]
    return {dataset_name: read_dataset(dataset_name) for dataset_name in dataset_names}

sdfs = extract()
```
Datasets now live in a single dictionary (`sdfs`), keyed by name, which simplifies downstream logic.

#### Splitting the logic into functional pieces
The core processing logic in the prototype notebook is implemented as a single chained PySpark query.
```python
sdf_prepared_data = (
    sdf_taxi_data
    .dropDuplicates()
    # keep only the evening rides in a time range
    .filter(F.date_format(F.col("pickup_datetime"), "yyyyMM") > "201703")
    .filter(F.date_format(F.col("pickup_datetime"), "yyyyMM") <= "201811")
    .filter(F.date_format(F.col("dropoff_datetime"), "HH") >= "17")
    .filter(F.date_format(F.col("dropoff_datetime"), "HH") <= "23")
    # remove rides to and from the airport
    .join(
        sdf_taxi_zone_geo
        .withColumnRenamed("zone_id", "pickup_location_id"),
        on="pickup_location_id"
    )
    .withColumnRenamed("zone_name", "pickup_zone_name")
    .withColumnRenamed("borough", "pickup_borough")
    .filter(~F.lower(F.col("pickup_zone_name")).like("%airport%"))
    .join(
        sdf_taxi_zone_geo
        .withColumnRenamed("zone_id", "dropoff_location_id"),
        on="dropoff_location_id"
    )
    .withColumnRenamed("zone_name", "dropoff_zone_name")
    .withColumnRenamed("borough", "dropoff_borough")
    .filter(~F.lower(F.col("dropoff_zone_name")).like("%airport%"))
    # take the first 3 rides per day for each pickup location
    .withColumn(
        "row_number",
        F.row_number()
        .over(
            Window
            .partitionBy(
                "pickup_location_id",
                F.date_format(F.col("pickup_datetime"), "ddMMyyyy")
            )
            .orderBy(F.asc("pickup_datetime"))
        )
    )
    .filter(F.col("row_number") < 4)
    # features engineering
    .withColumn("month", F.month(F.col("pickup_datetime")))
    .withColumn("day_of_week", F.dayofweek(F.col("pickup_datetime")))
    .withColumn("day_of_month", F.dayofmonth(F.col("pickup_datetime")))
    .withColumn(
        "store_and_fwd_flag",
        F.when(F.col("store_and_fwd_flag") == "N", 0)
        .otherwise(1)
    )
    .withColumn("random_number", F.rand())
)
```

While concise, it hides intent and makes modification risky.

We group related filters into functions and expose their parameters explicitly:
```python
history_start_month = "201703"
history_end_month = "201811"

def limit_history_to_a_range(sdf: DataFrame) -> DataFrame:
    pickup_month = F.date_format(F.col("pickup_datetime"), "yyyyMM")
    return (
        sdf
        .filter(pickup_month > history_start_month)
        .filter(pickup_month <= history_end_month)
    )
```

Parameters are kept outside functions to make them easy to override later (e.g., via runtime configuration).

#### Chaining logic with `.transform()`
Instead of creating a new dataframe variable for each step, we use `.transform()` (available in Spark 2.x+):
```python
sdf_prepared_data = (
    sdfs["taxi_trip_data"]
    .dropDuplicates()
    .transform(limit_history_to_a_range)
    .transform(keep_evening_rides_only)
    ...
)
```

This pattern keeps the pipeline linear, readable, and easy to modify during debugging.
The `.transform()` method can also take additional parameters:
```python
def exclude_airports_by_location(sdf: DataFrame, location_id_col_name: str) -> DataFrame:
    sdf_zone_geo_airport = (
        sdfs["taxi_zone_geo"]
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
```

Thus, we can reuse the function for both the pickup and drop off locations:
```python
sdf_prepared_data = (
    sdfs["taxi_trip_data"]
    ...
    .transform(exclude_airports_by_location, "pickup_location_id")
    .transform(exclude_airports_by_location, "dropoff_location_id")
    ...
)
```

#### Grouping logic into abstraction levels
Filtering and feature engineering form logical units in the transformation step of the code:
```python
def filter_data(sdf: DataFrame) -> DataFrame:
    return (
        sdf
        .dropDuplicates()
        .transform(limit_history_to_a_range)
        .transform(keep_evening_rides_only)
        .transform(exclude_airports_by_location, "pickup_location_id")
        .transform(exclude_airports_by_location, "dropoff_location_id")
        .transform(keep_first_n_daily_rides_only)
    )

def add_features(sdf: DataFrame) -> DataFrame:
    return (
        sdf
        .withColumn("month", F.month(F.col("pickup_datetime")))
        .withColumn("day_of_week", F.dayofweek(F.col("pickup_datetime")))
        .withColumn("day_of_month", F.dayofmonth(F.col("pickup_datetime")))
        .withColumn(
            "store_and_fwd_flag",
            F.when(F.col("store_and_fwd_flag") == "N", 0)
            .otherwise(1)
        )
    )

def transform() -> None:
    sdfs["prepared_data"] = (
        sdfs["taxi_trip_data"]
        .transform(filter_data)
        .transform(add_features)
    )
```

At this point, the executable logic shrinks to an ETL‑like skeleton:
```python
sdfs = extract()
transform()
model = train()
validate()
load()
```

### Step 1.2. Add logging
`print()` statements are acceptable in prototypes but not in production.
Replacing them with logging early avoids churn later.
A minimal setup below would allow for keeping the printed output while replacing the `print()` statements with `logger.info()`:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tip_amount_model_logger')
```

An extended setup allows consistent formatting and future redirection to files or logging systems:
```python
import logging

logger = logging.getLogger("tip_amount_model_logger")
logger.setLevel(logging.DEBUG) # lowest level to capture by the logger

logger.handlers.clear() # remove existing handlers to not accidentally duplicate them
sh = logging.StreamHandler() # handler for printing messages to console. Will need file handler in prod

sh.setLevel(logging.INFO) # lowest level for the handler to display
f = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M")
sh.setFormatter(f)
logger.addHandler(sh)

logger.info("Tip amount model logger is initialized!")
```

Notebook with the refactored code can be found [here](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/prototype_refactored.ipynb).

## Stage 2. Code in modules
*Suitable for code maintained by more than one person.*

Notebooks are convenient for exploration but awkward for collaboration and version control.
Reading flow is inverted (low‑level functions first) and merge request diffs are noisy.

Moving code into Python modules improves readability, collaboration, and testability.
I added the functionality to `.py` files in the `src` directory of [the repo](https://github.com/akoryachko/blog_posts/tree/main/pyspark_to_production).

### Step 2.1. Create the main module
Most logic lives in `tip_amount_model.py`.

Shared parameters are grouped into a configuration dataclass:
```python
from dataclasses import dataclass

@dataclass
class TipAmountModelConfig:
    history_start_month: str = "201703"
    history_end_month: str = "201811"
    first_evening_hour: str = "17"
    last_evening_hour: str = "23"
    n_first_daily_rides_to_keep: int = 3
    test_fraction: float = 0.2
    feature_cols: list[str] = [
        "passenger_count",
        "trip_distance",
        "rate_code",
        "store_and_fwd_flag",
        "payment_type",
        "fare_amount",
        "tolls_amount",
        "imp_surcharge",
        "month",
        "day_of_week",
        "day_of_month",
    ]
```

Another class holds the logic and shared variables to enable the computations:
```python
class TipAmountModel():
    def __init__(self, config: TipAmountModelConfig) -> None:
        self.config = config
        self.spark = SparkSession.builder.getOrCreate()
        self.sdfs = {}
        self.model = None
        self.feature_cols = feature_cols
```

A single `run()` method defines the table of contents for the job:
```python
class TipAmountModel():
    ...
    def run(self) -> None:
        self.extract()
        self.transform()
        self.train()
        self.validate()
        self.load()
```

Lower‑level methods follow immediately after their callers, preserving top‑down readability:
```python
    def extract(self) -> None:
        logger.info("Extracting datasets")
        dataset_names = [
            "taxi_trip_data",
            "taxi_zone_geo",
        ]
        for dataset_name in dataset_names:
            self.read_dataset(dataset_name)

    def read_dataset(self, dataset_name: str) -> None:
        file_path = f"../data/{dataset_name}.csv"
        self.sdfs[dataset_name] = self.spark.read.csv(
            file_path, header=True, inferSchema=True
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
            sdf
            .dropDuplicates()
            .transform(self.limit_history_to_a_range)
            .transform(self.keep_evening_rides_only)
            ...
        )

    def limit_history_to_a_range(self, sdf: DataFrame) -> DataFrame:
        ...
    
    def keep_evening_rides_only(self, sdf: DataFrame) -> DataFrame:
        ...
```


### Step 2.2. Supporting modules
Reusable infrastructure such as logging deserves [its own module](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/src/log_config.py).
A shared logging setup ensures consistent conventions across jobs by calling:
```python
from pyspark_to_production.src.log_config import get_logger
logger = get_logger(__name__)
```

Any function used across jobs should follow the logging module suite and be imported as opposed to defined within a job module.
One caveat with independent functions is that all parameters for such a function should be passed instead of being available through `self`.

### Step 2.3. Notebook interaction
Even with production code in modules, notebooks remain useful for debugging and experimentation.

#### 2.3.1. Debugging
Debugging module-based code from a notebook is primarily about **shortening the feedback loop** between code changes and observed behavior.

The typical debugging workflow consists of three steps:
1. Make the project modules importable.
1. Instantiate the job and run it.
1. Inspect intermediate state and iterate.

**Make modules available for import**  
Because the notebook lives outside the src directory, Python does not automatically know where to find the project modules.
One simple way to fix this in a notebook is to add the project root to Python’s import path:
```python
import sys
sys.path.append('../..')
```
Here we add the `blog_posts directory` (two levels above `notebooks`) to Python’s module search path, making `pyspark_to_production` importable.

**Create job objects**  
Next, import the main classes and create the configuration and job instances:
```python
from pyspark_to_production.src.tip_amount_model import TipAmountModelConfig, TipAmountModel
config = TipAmountModelConfig()
job = TipAmountModel(config)
```
At this point:
- All configuration parameters are accessible via `job.config`.
- The Spark session and internal state live inside the `job` object.

**Run the job**  
Finally, run the full pipeline to ensure that everything works end-to-end:
In our case it is:
```python
job.run()
```
If an error occurs, the stack trace will point directly to the failing method in the module code, making it clear where to investigate.

**Inspect intermediate results**  
One advantage of the class-based design is that intermediate artifacts are preserved on the job instance.
Any dataset stored in `self.sdfs` can be inspected interactively.

For example, to inspect the training dataset:
```python
job.sdfs["training"].show(5)
```

**Iterate on fixes**  
After fixing a bug in the module code, you have two options:
1. **Restart the notebook.**
This guarantees that all module changes are picked up, but it is slower.
2. **Enable automatic reloading.**
Jupyter can automatically reload modified modules before each execution by using magic commands:
```python
%load_ext autoreload
%autoreload 2
```

#### 2.3.2 Experimentation
One of the main advantages of interacting with the job through a notebook is the ability to experiment quickly while still using production-ready code.
Experimentation typically falls into two categories: parameter tuning and prototyping new logic.

**Parameter-level experimentation**
Many experiments can be performed by simply modifying configuration values and rerunning the relevant stages.
This is the safest and fastest way to explore model behavior, as it requires no code changes.

For example, to evaluate the impact of a smaller test set, update the configuration directly on the job instance and rerun the training and validation stages:
```python
job.config.test_fraction = 0.01
job.train()
job.validate()
```
Because the pipeline stages are independent, only the affected parts need to be rerun.
This makes parameter tuning fast and encourages systematic experimentation.

**Prototyping new logic**
Some experiments go beyond parameter changes and require trying out new ideas that are not yet part of the production code.
For example, you may want to evaluate a different model architecture or feature-processing strategy.
In such cases, the goal is to prototype **without modifying the repository**, keeping experiments local to the notebook.

Suppose we want to try a Gradient-Boosted Tree regressor instead of the default model.
We can define a modified training function directly in a notebook cell:
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor

def train_model(self) -> None:
    assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")

    gbt = GBTRegressor(
        labelCol="tip_amount",
        featuresCol="features",
        predictionCol="prediction",
        stepSize=0.1,
        maxDepth=4,
        featureSubsetStrategy="auto",
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, gbt])

    self.model = pipeline.fit(self.sdfs["training"])

    print("Modified content")
```

The function can then be dynamically bound to the existing job instance:
```python
import types
job.train_model = types.MethodType(train_model, job)
```

After that, the modified logic can be exercised by rerunning the relevant stages:
```python
job.train()
job.validate()
```

This approach allows rapid exploration of alternative implementations while preserving a clean separation between experimental code and the production codebase.
Once an experiment proves valuable, the changes can be formalized and properly integrated into the module.

All examples shown in this section are available in the [playground notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/playground.ipynb).


## Stage 3. Unit tests
*Required for time‑critical or frequently changing systems.*

Code changes introduce bugs far more often than we would like.
The goal of unit testing is to catch those bugs before they reach production, reducing rollbacks, hot fixes, and firefighting.

### Step 3.1. Checks in a notebook cell
The fastest way to start unit testing is to write simple behavioral checks directly in the module-interaction notebook from the previous stage.
As an example, let’s verify that the `add_features()` function actually produces the feature columns listed in `feature_cols`:
```python
from datetime import datetime

def is_subset(a: list, b: list) -> bool:
    return set(a) <= set(b)

data = [
    (datetime(2021, 1, 1, 12, 0, 0), "Y"),
    (datetime(2021, 6, 15, 9, 30, 0), "N")
]

expected_columns = job.feature_cols[-4:]

sdf_fake_input = job.spark.createDataFrame(
    data, schema=["pickup_datetime", "store_and_fwd_flag"]
)
assert not is_subset(expected_columns, sdf_fake_input.columns)

sdf_fake_features = job.add_features(sdf_fake_input)
assert is_subset(expected_columns, sdf_fake_features.columns)
```

This test:
- creates minimal fake input data,
- verifies that feature columns are not present before transformation,
- and confirms that they appear after `add_features()` is applied.

Even simple checks like this are effective at catching accidental column renames or mismatches between transformation logic and configuration.

### Step 3.2. Function in a notebook cell
While inline notebook checks work, they quickly become problematic:
- they reuse shared state (job) across tests,
- fake data generation is duplicated,
- schema changes require editing multiple cells.

To address this, we refactor the test logic into self-contained test functions and introduce reusable fake data generators.

**Reusable fake data generation**  
We start by defining lightweight schema classes and a helper function to generate PySpark Row objects with defaults:
```python
from pyspark.sql import Row
from datetime import datetime, timezone
from typing import TypeVar, Type
from dataclasses import dataclass, asdict

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
    data_class: Type[CT],
    data: list[tuple] = [()],
    columns: list[str] = []
) -> list[Row]:
    generated_rows = []
    for record in data:
        record_dict = dict(zip(columns, record))
        record_class = data_class(**record_dict)
        record_row = Row(**asdict(record_class))
        generated_rows.append(record_row)
    return generated_rows
```

This setup allows each test to override only the fields it cares about, while the rest are filled with sensible defaults.

**Test function**  
With reusable data generation in place, we can write proper unit tests that exercise larger parts of the pipeline.
The test function will then look like the follows:
```python
from datetime import datetime

def is_subset(a: list, b: list) -> bool:
    return set(a) <= set(b)

def test_add_features_column_names() -> None:

    tip_model = TipAmountModel(TipAmountModelConfig())

    columns=["pickup_datetime", "store_and_fwd_flag"]
    data = [
        (datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc), "Y"),
        (datetime(2021, 6, 15, 9, 30, 0, tzinfo=timezone.utc), "N"),
    ]

    tip_model.sdfs["taxi_trip_data"] = tip_model.spark.createDataFrame(
        generate_rows(Trip, data, columns)
    )
    tip_model.sdfs["taxi_zone_geo"] = tip_model.spark.createDataFrame(
        generate_rows(ZoneGeo)
    )

    assert not is_subset(tip_model.feature_cols, tip_model.sdfs["taxi_trip_data"].columns)

    tip_model.transform()
    assert is_subset(tip_model.feature_cols, tip_model.sdfs["prepared_data"].columns)
```

This test verifies behavior at the level of the `transform()` stage, which is why we also provide a minimal fake `taxi_zone_geo` dataset.

**Additional tests**  

The data-generation boilerplate becomes worthwhile once multiple tests reuse it.
For example, we can validate that airport trips continue to be excluded:
```python
def test_exclude_airports_by_location() -> None:
    columns=["pickup_location_id", "dropoff_location_id"]
    data = [
        (1, 1),
        (100, 1),
        (1, 100),
        (100, 100),
    ]

    tip_model = TipAmountModel(TipAmountModelConfig())

    # no airports
    tip_model.sdfs["taxi_trip_data"] = tip_model.spark.createDataFrame(
        generate_rows(Trip, data, columns)
    )
    tip_model.sdfs["taxi_zone_geo"] = tip_model.spark.createDataFrame(
        generate_rows(ZoneGeo, [(100, "terrestrial", ), ["zone_id", "zone_name"]])
    )

    tip_model.transform()
    assert tip_model.sdfs["prepared_data"].count() == 3

    # all except one have airports
    tip_model.sdfs["taxi_zone_geo"] = tip_model.spark.createDataFrame(
        generate_rows(ZoneGeo, [(100, "is airport or so", )], ["zone_id", "zone_name"])
    )

    tip_model.transform()
    assert tip_model.sdfs["prepared_data"].count() == 0
```

Test functions can be executed directly from a notebook cell:
```python
test_add_features_column_names()
test_exclude_airports_by_location()
...
```

This approach keeps the feedback loop short while building tests that can later be moved into a proper test suite with minimal changes.

All examples in this section are available in the [test notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/test.ipynb).

### Step 3.3. Testing modules
Notebook-based tests are far better than having no tests at all.
They work well for small projects or exploratory code, but they do not scale to large, actively developed, production-facing systems.

As the codebase grows, testing must be automated. A scalable testing setup should:
- **Run the full test suite.**
This provides a complete picture of what works and what breaks.
Notebook execution stops at the first failing assertion, hiding downstream failures.
- **Run fast.**
Restarting and rerunning notebooks during debugging quickly becomes expensive.
Tests should execute with minimal overhead to support rapid iteration.
- **Require minimal manual setup.**
Tests should not depend on a developer running them locally.
Instead, they should run automatically as part of a CI pipeline before changes are merged into the production branch.
Notebook creation and output inspection add unnecessary complexity in this context.

To achieve this, the notebook-based test functions are moved into test modules.
In this example, the tests are placed in `tests/unittests/test_tip_amount_model.py`.
The tests themselves remain largely unchanged.
The example uses `pytest`, while the standard `unittest` library could be used as well.

**Sharing a Spark session across tests**  
First, let's define a single shared Spark session for all the tests because recreating it for every test is slow and wasteful.
Adding a `pytest` fixture with session-level scope to the global settings module `conftest.py` will do the trick:
```python
from collections.abc import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator:
    spark: SparkSession = (
        SparkSession.builder.master("local[*]")
        .appName("testing")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.sparkContext.stop()
```

An additional benefit of this set up is that tests can now depend on the Spark session directly, instead of pulling it from a job instance.
The test signature changes accordingly:
```python
def test_add_features_column_names(spark: SparkSession) -> None:
    ...
    tip_model.sdfs["taxi_trip_data"] = spark.createDataFrame(generate_rows(Trip, data, columns))
    ...
```

**Parameterizing tests** 
`pytest` also allows running the same test logic with multiple input combinations.
This is especially useful for validating edge cases and understanding exactly which inputs cause failures.
For example, airport filtering can be tested as follows:
```python
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
    columns=["pickup_location_id", "dropoff_location_id"]
    data = [(pickup_location_id, dropoff_location_id), ]
    ...
    tip_model.transform()
    assert tip_model.sdfs["prepared_data"].count() == n_expected_rows
```

Here we test each combination of pickup and dropoff locations separately and also passing how many rows we expect after the filtering.

**Running the test suite**
Before running the tests, ensure that `__init__.py` files are present in both the project root and test directories so that Python can correctly resolve imports.

Tests can then be executed from the project root with:
```bash
python -m pytest
```
Optional flags:
- `-vv` to display test names;
- `-s` to show print statements and logs.

**Closing the loop**

Once the test infrastructure is in place, additional test functions should be added to cover the rest of the pipeline logic.
Tests should be updated alongside code changes and run automatically as part of the development workflow.

## Conclusion
Notebooks are an excellent medium for exploration, but they are a poor long-term container for production logic.
The problem is not the use of notebooks themselves, it is the lack of a clear path from experimentation to maintainable code.

In this post, we walked through that path step by step:
- Starting from a working but monolithic PySpark notebook,
- Refactoring logic into readable, composable functions,
- Moving code into modules with clear ownership and structure,
- Preserving the notebook as a tool for debugging and experimentation,
- And, finally, introducing automated tests to protect production behavior.

Each stage represents a trade-off between speed and rigor.
Not every project needs to reach the final stage, but every project benefits from knowing when to stop.
A one-off analysis, a recurring internal report, and a business-critical pipeline all justify different levels of structure and testing.

The key idea is not to “productionize everything”, but to design notebooks and modules so that evolution is possible.
When code is structured with clear stages, explicit parameters, and testable units, the handoff between data scientists and engineers becomes collaboration instead of reimplementation.
