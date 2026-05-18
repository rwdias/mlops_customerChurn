import time
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

class DataProcessing:

    def __init__(self, spark_df: SparkDataFrame, config: ProjectConfig, spark: SparkSession) -> None:
    self.df = spark_df  # Store the Spark DataFrame as self.df
    self.config = config  # Store the configuration
    self.spark = spark