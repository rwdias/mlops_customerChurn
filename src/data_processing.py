from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, avg
from pyspark.sql.types import BooleanType, IntegerType, DoubleType, StringType, FloatType
from databricks.feature_engineering import FeatureEngineeringClient
from typing import Tuple
from config import ProjectConfig
from data_utils import calculate_missing

# config = ProjectConfig.from_yaml(config_path="../project_config_telcochurn.yml", env="dev")

class DataProcessing:

    def __init__(self, dataset_path: str, config: ProjectConfig, spark: SparkSession) -> None:
        self.spark = spark 
        self.config = config
        
        self.df = (
            self.spark.read
            .option("nullValue", " ")
            .csv(dataset_path, header=True, multiLine=True, escape="'", inferSchema=True)
        )

    def preprocess(self) -> None:
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        binary_features = self.config.binary_features
        self.fe = FeatureEngineeringClient()
        target = self.config.target
        per_thresh = 0.6

        # 1. Deduplicação e seleção de colunas para evitar o erro do Spark Connect
        colunas_unicas = list(set(cat_features + num_features + [target]))
        self.df = self.df.select(colunas_unicas)

        # 2. Conversão da coluna tenure para Double de forma isolada
        if "tenure" in self.df.columns:
            self.df = self.df.withColumn("tenure", col("tenure").cast(FloatType()))

        # 3. Mapeamento seguro de todas as colunas binárias de Texto/Int para Double
        for feature in binary_features:
            if feature in self.df.columns:
                # Converte para String dentro do 'when' para aceitar qualquer entrada (Int, String ou Bool) de forma universal
                feature_str = col(feature).cast(StringType())
                
                self.df = self.df.withColumn(
                    feature,
                    when((feature_str == "Yes") | (feature_str == "yes") | (feature_str == "true") | (feature_str == "True") | (feature_str == "1"), 1.0)
                    .otherwise(0.0)
                    .cast(FloatType())
                )

        # 4. Label Encoding para as variáveis ternárias textuais remanescentes
        ternary_features = ["MultipleLines", "TechSupport"]
        for feature in ternary_features:
            if feature in self.df.columns:
                self.df = self.df.withColumn(
                    feature,
                    when((col(feature) == "Yes") | (col(feature) == "yes"), 2.0)
                    .when((col(feature) == "No") | (col(feature) == "no"), 1.0)
                    .otherwise(0.0).cast(FloatType())  
                )

        # 5. One-Hot Encoding Manual estritamente para a coluna 'Contract'
        categories_mapping = {
            "Contract": ["Month-to-month", "One year", "Two year"]
        }

        for column, categories in categories_mapping.items():
            if column in self.df.columns:
                for category in categories:
                    clean_category_name = category.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    ohe_column_name = f"{column}_OHE_{clean_category_name}"
                    
                    self.df = self.df.withColumn(
                        ohe_column_name,
                        when(col(column) == category, 1.0).otherwise(0.0)
                    )

        # 6. Remoção de registros com TotalCharges nulos ou inconsistentes (se a coluna existir)
        if "TotalCharges" in self.df.columns:
            self.df = self.df.filter(
                (col("TotalCharges") > 0) | 
                (col("TotalCharges").isNull())
            )
        
        # 7. Tratamento de valores ausentes por Limiar (Drop de colunas vazias)
        missing_df = calculate_missing(self.df, self.spark)
        N = self.df.count()
        
        to_drop_missing = [
            x.asDict()['Column'] for x in missing_df.select("Column")
            .where(col("Number of Missing Values") / N >= per_thresh).collect()
        ]
        print(f"Dropping columns {to_drop_missing} for more than {per_thresh * 100}% missing data")
        self.df = self.df.drop(*to_drop_missing)

        # 8. Imputação segura de dados nulos (Numéricos) - Executado após conversões de tipo
        num_cols = [c.name for c in self.df.schema.fields if (c.dataType == DoubleType() or c.dataType == IntegerType())]
        self.df = self.df.na.fill(value=0.0, subset=num_cols)

        # 9. Imputação de dados nulos (Booleanos)
        bool_cols = [c.name for c in self.df.schema.fields if (c.dataType == BooleanType())]
        self.df = self.df.na.fill(value=False, subset=bool_cols)

        # 10. Imputação de dados nulos (Strings residuais, ex: Contract original)
        to_exclude = ["customerID", "Contract"]
        string_cols = [c.name for c in self.df.drop(*to_exclude).schema.fields if c.dataType == StringType()]
        self.df = self.df.na.fill(value='No', subset=string_cols)
        
        # 11. REORDENAÇÃO DA CHAVE PRIMÁRIA (Garante customerID em primeiro)
        if "customerID" in self.df.columns:
            # Coloca customerID no início e adiciona todas as outras colunas depois
            outras_colunas = [c for c in self.df.columns if c != "customerID"]
            self.df = self.df.select(["customerID"] + outras_colunas)
    
    def split_data(self, train_size: float = 0.8, seed: int = 42) -> Tuple[SparkDataFrame, SparkDataFrame]:
        """Split the Spark DataFrame (self.df) into training and test sets."""
        test_size = 1.0 - train_size
        weights = [train_size, test_size]
        
        if "customerID" in self.df.columns:
            df_stable = self.df.sort("customerID")
        else:
            df_stable = self.df
            
        train_set, test_set = df_stable.randomSplit(weights, seed=seed)
        return train_set, test_set
    
    def save_feature_tables(self, df_train: SparkDataFrame, df_test: SparkDataFrame) -> None:
        """Saves train and test dataframes as Databricks Feature Store tables 
        using catalog and schema dynamically resolved from the YAML config.
        Ensures Unity Catalog primary key constraints are properly created.
        """
        catalog = self.config.catalog_name
        schema = self.config.schema_name
        
        tables_config = {
            f"{catalog}.{schema}.telco_features_train": df_train,
            f"{catalog}.{schema}.telco_features_test": df_test
        }
        
        for table_name, df_set in tables_config.items():
            try:
                # Se a tabela já existe, mas foi criada sem PK pelo Spark, o create_table falhará.
                # Para garantir o overwrite limpo com a PK, dropamos a tabela existente se houver.
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")
                
                # Cria e registra a tabela do zero no Feature Store com a constraint de PK necessária
                self.fe.create_table(
                    name=table_name,
                    primary_keys=["customerID"],
                    df=df_set,
                    description="Telco customer features",
                    tags={"source": "bronze", "format": "delta"}
                )
                print(f"[{catalog}] Tabela de Features '{table_name}' criada e registrada com sucesso.")
                
            except Exception as e:
                # Fallback de segurança caso a tabela esteja bloqueada ou o drop não seja ideal
                if "already exists" in str(e):
                    self.fe.write_table(
                        name=table_name,
                        df=df_set
                    )
                    print(f"[{catalog}] Dados da tabela '{table_name}' sobrescritos via write_table.")
                else:
                    raise e
