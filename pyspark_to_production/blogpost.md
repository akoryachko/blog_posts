# Convert PySpark notebook to production code

## Motivation
The gap between a working notebook and a maintainable production code is significant especially in teams where everyone specializes on a separate part of the project lifecycle.
Cases when data scientists throw a proof of concept notebook over the fence to engineers are common.
Engineers take the logic as is, wrap it up as a job, and schedule the run.
This approach creates solutions that are hard to understand, modify, and maintain.
The issue is exacerbated in big data applications where data scientists lack coding skills while engineers do not feel confident with untangling PySpark queries.

## Purpose
This post concentrates on practices to bridge the gap between a free flow PySpark notebook and a modular code ready for a production repo.
The following tips and tricks should help data scientists develop a more readable and reusable code and give engineers courage to refactor PySpark queries in a way that agrees with coding standards.

## Approach
We will go through an example notebook that trains a model for predicting a taxi ride tip amount.
The majority of the logic in the notebook is in a long PySpark query that does the job but might be hard to understand even for its creator after some time.
I will break the logic down into functions and talk about tricks for making the code more readable and reusable.

Each project has its own maintainability requirements: from none for a one-and-done analysis to a lot for a critical constantly updated system with strict service level agreements (SLAs).
The blogpost progresses through the stages of improvements that align with the requirements severity.
Feel free to stop the implementation of suggestions after any stage if all the requirements got satisfied.

## Example project
Let's assume that we have the following artificial task at hand.
We want to train a model that predicts a tip amount for New York taxi rides based on other information about the trip.
We are interested in the first 3 evening rides only and want to exclude rides to and from the airport.
The model should be updated on a regular basis.
We should have a way to check feature importance and evaluation metrics for each run.

## Stage 0. Notebook solution
*Suitable only for one time analysis or proof of concept projects.*

The full notebook containing the solution code can be found [here](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/prototype.ipynb).

Check [`README.md`](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/README.md) to set up environment for running the notebook locally.

## Stage 1. Refactored notebook
*Suitable for code that needs to be run at rare occasions.*

Once we made sure the notebook produces the result for a single run, it is time to make it pretty.
The goal is to make the code more readable and the execution more trackable.

### Step 1.1. Put code in functions
This step aims to put the code in a structure that is easy to navigate and change.
We achieve that by breaking the code into functions with giving meaningful names and execute those functions in order.

We will concentrate on the main processing logic listed below first.

```python
import pyspark.sql.functions as F
from pyspark.sql import Window

sdf_taxi_data = spark.read.csv("../data/taxi_trip_data.csv", header=True, inferSchema=True)
sdf_taxi_zone_geo = spark.read.csv("../data/taxi_zone_geo.csv", header=True, inferSchema=True)

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

# train test split
sdf_training = sdf_prepared_data.filter(F.col("random_number") > 0.2)
sdf_test = sdf_prepared_data.filter(F.col("random_number") <= 0.2)
```

The implementation has a number of issues that needs to be addressed.

1. **Repeating pieces of code.**
We will have to change the code in more than one place if, for example, the path to the folder with the datasets changes.
1. **Magic numbers.**
The meaning of `"032017"`, `"23"`, `50`, and other values is intuitive at the time of development but will lead to a lot of confusion when we get back to the code, say, a year from now.
1. **Rotting comments.**
The comments separate blocks of logic and create a feeling of a structure.
However, comments are updated at a far lower pace than the code itself.
So future code modifications will make comments obsolete and become misleading pretty quickly.

Let's start improving the structure with a function for loading the datasets.
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
Here we resolved the code duplication and made it easy to add/remove the datasets.
The datasets will stay together in the dictionary of datasets (`sdfs`) with keys corresponding to dataset file names.

Next, let's put the time range filtering functions together.
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

first_evening_hour = "17"
last_evening_hour = "23"

def keep_evening_rides_only(sdf: DataFrame) -> DataFrame:
    dropoff_hour = F.date_format(F.col("dropoff_datetime"), "HH")
    return (
        sdf
        .filter(dropoff_hour >= first_evening_hour)
        .filter(dropoff_hour <= last_evening_hour)
    )
```

Here we removed code duplication and explained the magic numbers.
I am keeping some parameters declarations outside of the functions to make it easier to pass those from outside as runtime settings at a later stage.
I intentionally removed comments to force descriptive function and variables names that shouldn't rot as fast.

The best part of this setup is how these functions can be applied to a dataset.
I would argue against creating a separate dataset for each function call.
Use the `.transform` method instead to chain the processing stages.
This way the first 2 steps of processing will turn into something like this:
```python
sdf_prepared_data = (
    sdfs["taxi_trip_data"]
    .dropDuplicates()
    .transform(limit_history_to_a_range)
    .transform(keep_evening_rides_only)
    ...
)
```

The beauty of using the `.transform` is that you won't need to come up with dataframe names while keeping the ability to turn on and off pieces of logic when debugging by commenting lines out.
The `.transform` method can take parameters in addition to the function name as shown below for functions that excludes airports as either trip source or destination.
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

We will filter airports from the pickup and drop off locations as follows:
```python
sdf_prepared_data = (
    sdfs["taxi_trip_data"]
    ...
    .transform(exclude_airports_by_location, "pickup_location_id")
    .transform(exclude_airports_by_location, "dropoff_location_id")
    ...
)
```

The next step of keeping the first 3 rides only follows the same logic:
```python
n_first_daily_rides_to_keep = 3

def keep_first_n_daily_rides_only(sdf: DataFrame) -> DataFrame:
    pickup_date = F.date_format(F.col("pickup_datetime"), "yyyyMMdd")
    window = (
        Window
        .partitionBy("pickup_location_id", pickup_date)
        .orderBy(F.asc("pickup_datetime"))
    )
    return (
        sdf
        .withColumn("ride_number", F.row_number().over(window))
        .filter(F.col("ride_number") <= n_first_daily_rides_to_keep)
        .drop("ride_number")
    )
```

Feature engineering transformations do not require a split into functions.
They are concise and self explanatory.
However, they form a logical block that can be abstracted away along with filtering transformations leading to the following structure:
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

def transform() -> PipelineModel:
    logger.info("Preparing the data")
    sdfs["prepared_data"] = (
        sdfs["taxi_trip_data"]
        .transform(filter_data)
        .transform(add_features)
    )
```

Wrapping model training and evaluation code into functions is straightforward and can be found in the [refactored notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/prototype_refactored.ipynb).
After that, the executable code shrinks to the definition of parameters and an ETL-like framework:
```python
sdfs = extract()
transform()
model = train()
validate()
load()
```

The refactored code makes it much easier to navigate.
A spot for bugfix or an improvement can be found by going from higher abstraction level functions to the lower ones. 

### Step 1.2. Add logging
We have a number of print statements that show model quality after a training.
While acceptable for POC, print statements are not suitable for production code and have to be replaced with logging.
So, it is easier to make a switch from printing to logging earlier than later while preserving the outcome.
The following set up along with changing `print` to `logger.info` will produce the same result:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tip_amount_model_logger')
```

An alternative extended boilerplate can be used to format the message and make it easier to change from the standard output to writing logs to a file later.

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

## Stage 2. Code in modules
*Suitable for code that gets updated by more than a single person.*

While good for prototyping, a notebook poses challenges for code readability and editing.
Think of a book or an article as examples of natural reading flow.
Each starts with the highest level abstraction like a title, and goes down through lower abstraction levels like chapters, sections, subsections, etc.
A notebook forces us to define the low level abstraction functions first to use them in the following cells thus breaking the readability flow and making it harder to navigate the code.
Editing is also challenging because tracking notebook changes is hard in a repo and almost impossible if it stays outside of it.
A modular solution will make the reading flow even better while facilitating collaboration on the functionality in the repo. Hence, the next step towards a production ready solution is to put the code in `.py` files which I added to the `src` directory.

### Step 2.1. Create the main module
The majority of code will go the the main module called `tip_amount_model.py`.

We used quite a number of parameters and variables across the functions.
Passing those parameters from the main function will blow up functions definitions.
A better way to share parameters across a number of logically connected functions is to put them in a class.
This way the shared parameters will be accessible through the `self` attribute.

The class definition will then look like follows:
```python
class TipAmountModel():
    def __init__(self, config: TipAmountModelConfig) -> None:
        self.config = config
        self.spark = SparkSession.builder.getOrCreate()
        self.sdfs = {}
        self.model = None
        self.feature_cols = feature_cols
```

Here we
- pass the configuration variables in a single variable,
- bind the class variable `spark` to an existing or newly created spark session,
- define other shared variables.

The configuration variables class makes it easier to run the job with different parameters.
A type validation for the passed parameters is typically a good idea, so a dataclass was preferred over, say, a dictionary:
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

Next, let's define the top abstraction level function that runs the whole script.
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

The `run()` function contains the table of contents for our script and is located just one required function away from the title.
The table of contents facilitates navigation across the script chapter by chapter.
Hence, the next function defined should be chapter 1 - `extract()`:
```python
    def extract(self) -> None:
        logger.info("Extracting datasets")
        dataset_names = [
            "taxi_trip_data",
            "taxi_zone_geo",
        ]
        for dataset_name in dataset_names:
            self.read_dataset(dataset_name)
```

The function has a number of modifications compared to the one in the refactored notebook:
- Takes `self` as an input parameter.
This allows to have an access to all class variables.
- Has nothing to return.
Class variables allow for assignment through `self`, so no need to pass things around.
- Allows to fill the dataframes dictionary outside of the function body by calling `read_dataset()` that shares the `self` variable.

The `extract` function calls a lower abstraction level function `read_dataset`.
This means that the `read_dataset` function definition should go next to not interrupt the flow.
Think of it in a manner similar to how section goes within the chapter.
```python
    def read_dataset(self, dataset_name: str) -> None:
        file_path = f"../data/{dataset_name}.csv"
        self.sdfs[dataset_name] = self.spark.read.csv(
            file_path, header=True, inferSchema=True
        )
```

As expected, the dataframes dictionary gets filled here.

This concludes the `extract` chapter, so we can move forward to the next one.
The changes for other methods are similar to the ones for the `extract` method and available in full in the [corresponding module](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/src/tip_amount_model.py).
Below is an extract showing that the general structure of the chapters followed by sections and subsections is still intact:
```python
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
    
    ...
```

### Step 2.2. Create supporting modules
The logger creation boilerplate deserves [its own module](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/src/log_config.py) because it standardizes the logging format and conventions.
Any function used across jobs should follow the logging module suite and be imported as opposed to defined:
```python
from pyspark_to_production.src.log_config import get_logger
logger = get_logger(__name__)
```
One caveat with independent functions is that all parameters for such a function should be passed instead of being available through `self`.

### Step 2.3. Interact with modules through a notebook
Having the code in functions across modules is beneficial for production.
However, it might scare Data Scientists away from working on the code due to not as intuitive way of interaction.
The following approach allows for debugging and modifying the module code from a notebook.

#### 2.3.1. Debugging
The first step to debugging a module in a notebook is making it available for import.
One of the ways to do that is to add the corresponding path to the list of paths python uses for importing the modules.
In our case something like that should work:
```python
import sys
sys.path.append('../..')
```
Here we add the `blog_posts` directory which is two levels higher than the `notebooks` folder to python's search path.

Next we import the required classes from the module and create the corresponding objects:
```python
from pyspark_to_production.src.tip_amount_model import TipAmountModelConfig, TipAmountModel
config = TipAmountModelConfig()
job = TipAmountModel(config)
```

Finally, run the main function to make sure everything is working.
In our case it is:
```python
job.run()
```

In case of bugs, the error messages will point where to look for issues.
Class parameters are available for inspection through the class instance.
For example, one of the intermediate datasets can be checked as follows:
```python
job.sdfs["training"].show(5)
```

Restart the notebook after the fixing the bugs in the module.
This way the changes take place for the next try.
Another option is to use the following Jupyter magic to make the update process automatic:
```python
%load_ext autoreload
%autoreload 2
```

#### 2.3.2 Trying out different parameter values
Processing stages can be rerun after modifying parameters directly.
For example, the training and validation stages can be rerun after reducing the test fraction to observe the impact:
```python
job.config.test_fraction = 0.01
job.train()
job.validate()
```

#### 2.3.3 Prototyping
Another reason for interacting with the module could be trying out new things outside of what is reachable with parameter modification.
For example, we want to try a different model.
This requires a code modification.
We don't want to modify the repo but just play around with the class instance.
Let's start with defining the modified function in a cell:
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

Then we need to bind this function to the class instance:
```python
import types
job.train_model = types.MethodType(train_model, job)
```

After that run the required stages to see the results.
```python
job.train()
job.validate()
```

All these examples can be found and run in [the playground notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/playground.ipynb).

## Stage 3. Unit tests
*Required for the code used in time critical applications.*

Code changes introduce bugs more often than not.
We want to find those bugs before they hit production and put it down.
Testing the functionality before shipping code changes to production should prevent a decent amount of rollbacks and hot fixes.

### Step 3.1. Test in a notebook cell

The easiest way to start unit testing is to write functionality checks in a cell of the module interaction notebook from the previous stage.
Let's make sure that the `add_features()` function produces the columns that we listed in `feature_cols`.
Code for testing that can look something like this:
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

The code creates fake data, makes sure that not all of the feature columns are present before applying the function, and that all of them are there after the function is applied.
This way, something like an accidental column renaming in the function or in the `feature_cols` would make this code raise an assertion error.

### Step 3.2. Function in a notebook cell
Raw code in a cell will work but it has a number of disadvantages such as the reuse of the `job` variable created once and modified from test to test and custom fake data generation that has to be adjusted in every test function if the input data schema changes.
We will refactor the test code in a function that independently creates all the variables and put together a fake data generator that can be reused by other functions.
The fake data generator would look something like the following:
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

Here we put together schemas for both datasets and a function that generates PySpark rows.
The `generate_rows()` function takes a dataset schema class, the row values for the columns that we want to modify, and the names of such columns.
Values for the columns that we pass will be accompanied with the default values for the rest of the columns within the dataset schema.

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

Here we test the functionality on a wider scope of the `transform` function.
That is why we faked the `taxi_zone_geo` dataset as well with a single row of default values.
First, we check that the features cols are not a part of the input data.
Then, we make sure that all of them appear after running the `transform` function.

The fake data generation boilerplate can not be justified by the use in a single test function.
Let's add another test function that makes sure that the airports keep being excluded.
The following function should do the trick:
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

These and other test functions can be run in a separate cell as such:
```python
test_add_features_column_names()
test_exclude_airports_by_location()
...
```

All this test code can be found and run in [the test notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/test.ipynb).

### Step 3.3. Testing modules
Code tests in a notebook is much better than no tests at all.
The approach can work for small and not likely to be modified projects but it will not scale.
Large active production facing projects have to have unit testing automated.
The solution should:
- **Run all of the tests.**
This gives the full picture of what is working and what is not.
Running in a notebook will fail at the first test that did not pass.
- **Run tests fast.**
Time for restarting and rerunning the notebook after each code change can accumulate during the complex debugging sessions.
- **Have a minimal setup.**
Testing should not depend on a person running them locally.
Testing should be a part of an automated pipeline that executes before the changes are merged to the production branch.
Notebook creation and output analysis overhead create an unnecessary complexity for the pipeline setup.

I used `pytest` library to put together and run the testing modules, but `unittest` library can do the trick as well.

First, let's put the test functions from the notebook to the module `test_tip_amount_model.py` in the `tests/unittests` folder of the project.

Next, let's create a common spark session for all the tests because recreating it for every test seems like a waste of resources.
This can be done by adding global settings to the `conftest.py` module:
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

Here the pytest decorator with the scope `session` will make sure to create and keep the spark variable around when running all the tests.
More about different scopes can be found [here](https://docs.pytest.org/en/6.2.x/fixture.html).
Another benefit of this fixture is that we can use spark directly instead of pulling it from a class instance.
This will change the test function definition as follows:
```python
def test_add_features_column_names(spark: SparkSession) -> None:
    ...
    tip_model.sdfs["taxi_trip_data"] = spark.createDataFrame(generate_rows(Trip, data, columns))
    ...
```

`pytest` can also run the same test with different parameters as different tests.
This would tell us which set of parameters broke the test instead of showing a general failure.
The effect can be achieved with another fixture as follows:
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

One last step before running the tests is putting `__init__.py` files to the top level and test folders, so that test modules can find the package modules.

Run the tests by typing something like `python -m pytest` in the terminal.
Optionally, append `-vv` option to see the names of the testing functions and/or `-s` option to see the prints/logs.

Once everything is up and running, test functions covering the rest of the functionality should be added to extend the tested code coverage. The tests should be regularly run and updated when introducing changes to the code.

## Conclusion
The blogpost presented tips and tricks for transforming the notebook code to a production level repo.
The list is not comprehensive but should help with breaking the silos between analytic and implementation parts of a development team with a purpose of creating readable, robust, and easy to change software solutions.
