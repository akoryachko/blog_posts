# Convert PySpark notebook to production code

## Motivation
The gap between a working notebook and a maintainable production code is significant especially in teams where everyone specializes on a separate part of the project lifecycle.
Cases when data scientists throw a proof of concept notebook over the fence to engineers are common.
Engineers take the logic as is, wrap it up as a job, and schedule the run.
This approach creates solutions that are hard to understand, modify, and maintain.
The issue is especially exacerbated in big data applications where data scientists lack coding skills while engineers do not feel confident with untangling PySpark queries.

## Purpose
This post concentrates on practices to bridge the gap between a free flow PySpark notebook and a modular code ready for a production repo.
The following tips and tricks should help data scientists develop a more readable and reusable code and give engineers courage to refactor PySpark queries in a way that agrees with coding standards.

## Approach
We will go through an example notebook that trains a model for predicting a taxi ride tip amount.
The majority of the logic in the notebook is in a long PySpark query that does the job but might be hard to understand even for its creator after some time.
I will break the logic down into functions and talk about tricks for making the code more readable and reusable.

## Example assignment
Let's assume that we have the following artificial task at hand.
We want to train a model that predicts a tip amount for New York taxi rides based on other information about the trip.
We are interested in the first 3 evening rides only, want to cap the price at $50, and exclude rides to and from the airport.
The model should be updated in a regular schedule with the new data at hand.
We should have a way to check feature importance and evaluation metrics for each run.

## Notebook
The full notebook with code can be found [here](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/notebook.ipynb).

## Refactoring
Once we made sure the notebook produces the result for a single run, it is time to make it pretty.

### Step 1. Put queries in functions
Below is the main processing logic that we are going to concentrate on first:

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
            .partitionBy("pickup_location_id", F.date_format(F.col("pickup_datetime"), "ddMMyyyy"))
            .orderBy(F.asc("pickup_datetime"))
        )
    )
    .filter(F.col("row_number") < 4)
    # features engineering
    .withColumn("month", F.month(F.col("pickup_datetime")))
    .withColumn("day_of_week", F.dayofweek(F.col("pickup_datetime")))
    .withColumn("day_of_month", F.dayofmonth(F.col("pickup_datetime")))
    .withColumn("store_and_fwd_flag", F.when(F.col("store_and_fwd_flag") == "N", 0).otherwise(1))
    .withColumn("random_number", F.rand())
)

# train test split
sdf_training = sdf_prepared_data.filter(F.col("random_number") > 0.2)
sdf_test = sdf_prepared_data.filter(F.col("random_number") <= 0.2)
```

The implementation has a number of issues that need to be addressed.

1. Repeating pieces of code.
We will have to change the code in more than one place if, for example, the path to the folder with the datasets changes.
1. Magic numbers.
The meaning of `"032017"`, `"23"`, `50`, and other values is intuitive at the time of development but will lead to a lot of confusion when we get back to the code, say, a year from now.
1. Rotting comments.
The comments separate blocks of logic and create a feeling of a structure.
However, comments are updated at a far lower pace than the code itself.
So future future code modifications will make comments obsolete and become misleading pretty quickly.

Let's start with creating a function for loading the datasets.
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
Here, we removed the code duplication and made it easy to add/remove the datasets and modify the path to datasets folder.
The access to the datasets will be through the dictionary of datasets (`sdfs`) with keys corresponding to dataset file names.

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

Here, we removed code duplication and explained the magic numbers.
I am keeping some parameters declarations outside of the functions to make it easier to pass those from outside as runtime settings at the later stage.
I intentionally removed comments to force descriptive function names that shouldn't rot as fast.

The best part of this setup is how these functions can be applied to a dataset.
I would argue against creating a separate dataset for each function call.
Use the `.transform` method instead.
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

The beauty of using the `.transform` is that you won't need to come up with dataframe names which will be similar to transformation functions anyway while keeping the ability to turn on and off pieces of logic when debugging by commenting lines out.
The `.transform` method can take parameters in addition to the function name as shown below for functions that excludes airports as either trip source or destination.
```python
def exclude_airports_by_location(sdf: DataFrame, location_id_col_name: str) -> DataFrame:
    sdf_zone_geo_no_airport = (
        sdfs["taxi_zone_geo"]
        .filter(~F.lower(F.col("zone_name")).like("%airport%"))
    )
    return (
        sdf
        .join(
            sdf_zone_geo_no_airport,
            on=[F.col(location_id_col_name) == F.col("zone_id")],
            how="leftsemi"
        )
    )
```

The function can be applied for the pickup and drop off locations as follows:
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

Feature engineering transformations are concise and self explanatory.
Putting a separate function for them will be a waste of effort.
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
        .withColumn("store_and_fwd_flag", F.when(F.col("store_and_fwd_flag") == "N", 0).otherwise(1))
    )

test_fraction = 0.2

def train_test_split(sdf: DataFrame) -> tuple[DataFrame]:
    return (
        sdf
        .randomSplit(
            weights=[1-test_fraction, test_fraction],
            seed=42
        )
    )
```

Wrapping model training and evaluation code into functions is straightforward and can be found in the [refactored notebook](https://github.com/akoryachko/blog_posts/blob/main/pyspark_to_production/notebooks/notebook_refactored.ipynb).
After that, the executable code shrinks to the definition of parameters and an ETL like framework:
```python
sdfs = extract()
model = transform()
validate()
load()
```


### Step 2. Add logging
We have a number of print statements that show model quality after a training.
While acceptable for POC, print statements are not suitable for production code and have to be replaced with logging.
The following set up along with changing `print` to `logger.info` will produce the same result:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tip_amount_model_logger')
```

This boilerplate can be further expanded to format the message and make it easier to change from the standard output to writing logs to a file later.

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

### Step 3. Put the code in modules
While good for prototyping, a notebook poses challenges for code readability and reusability.
For example, a natural flow of reading starts with the largest abstraction sections like title and chapter name if we take a book.
A notebook forces us to define the low level abstraction function first to use them in the following cells thus breaking the readability flow and making it hard to navigate the code.
Reusability also lacks because the functions defined in the notebook can not be easily reused in other notebooks.
Hence, the next step towards a production ready solution is to put the code in `.py` files located in `src` directory.

#### 3.1 Main module
The majority of code will go the the main module called `tip_amount_model.py`.

We used quite a number of parameters and variables across the functions.
Passing those parameters from the main function will blow up functions definitions.
A better way to share parameters across a number of logically connected function is to put them in a class.
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
- pass the configuration variables in one structure,
- create logger with the class name (`TipAmountModel`),
- bind the internal class variable to the existing or newly created spark session,
- define other shared variables.

The configuration variables class makes it easier to run the job with different parameters.
A type validation for the passed parameters is typically a good idea, so a dataclass is preferred over, say, a dictionary:
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
    def run(self):
        self.extract()
        self.transform()
        self.validate()
        self.load()
```

This is the table of contents for our script located just one required function away from the title.
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
- take `self` as an input parameter. This allows to have an access to all class variables;
- nothing to return. Class variables allow for assignment through `self` so no need to pass things around;
- the dataframes dictionary gets filled outside of the function through the `self` variable.

The `extract` function calls a lower abstraction level function `read_dataset`.
This means that the `read_dataset` function definition should go next to not interrupt the flow similar to how section goes within the chapter.
```python
    def read_dataset(self, dataset_name: str) -> None:
        file_path = f"../data/{dataset_name}.csv"
        self.sdfs[dataset_name] = self.spark.read.csv(file_path, header=True, inferSchema=True)
```

As expected, the dataframes dictionary gets filled here.
Though the decision about putting it here or in the higher level function depends on implementation as well as whether to keep `file_path` and `dataset_names` variables within the functions or pass them as a part of configuration structure.

This concludes the `extract` chapter, so we can move forward to the next one.
The changes are similar to the ones for the `extract` method and available in full in the [!!!!corresponding module!!!!](addalink).
Below is an extract showing that the general structure of chapters followed by sections and subsections is still intact:
```python
    def transform(self) -> None:
        logger.info("Preparing the data for training")
        self.prepare_data()
        ...
    
    def prepare_data(self) -> None:
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


<!-- 
Even the code creators forget what they meant with certain lines of code and have to rewrite things from scratch once requirements change.
Moreover, the same pieces of code keep being rewritten over and over causing longer development cycles and higher chances of errors.
I will use the standard output handler for the logger just to show that it can be used for printing things to the console.
However, having logs set in this way will make adding a file handler for the logs straightforward.

The minimum logger set up would look something like the following:

Handlers and formatters will increase the boilerplate but will also provide more control and an easier transition to logging into files:
-->