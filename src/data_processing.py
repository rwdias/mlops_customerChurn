import time
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

class DataProcessing:

    # 1. Altere o tipo do parâmetro para str (caminho do dataset)
    def __init__(self, dataset_path: str, spark: SparkSession) -> None:
        # 2. Defina o self.spark PRIMEIRO
        self.spark = spark 
        
        # 3. Agora self.spark está disponível para fazer a leitura
        self.df = (
            self.spark.read
            .option("nullValue", " ")
            .csv(dataset_path, header=True, multiLine=True, escape="'", inferSchema=True)
        ) 

    def display(self) -> None:
        # Nota: Se estiver no Databricks, display() funciona. 
        # Caso contrário, utilize self.df.show()
        display(self.df)
        
        