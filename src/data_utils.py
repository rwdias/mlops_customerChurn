from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, count, trim, lower
from pyspark.sql.types import StringType


def calculate_missing(
    input_df: DataFrame,
    spark: SparkSession,
    show: bool = True
) -> DataFrame:
    """
    Calcula quantidade e percentual de missing values por coluna.

    Parameters
    ----------
    input_df : DataFrame
        DataFrame Spark de entrada.
    spark : SparkSession
        Sessão Spark usada para criar o DataFrame de saída.
    show : bool, default=True
        Se True, exibe o resultado ordenado por quantidade de missing.

    Returns
    -------
    DataFrame
        DataFrame Spark com colunas:
        - Column
        - Number of Missing Values
        - Missing Percentage
    """

    total_rows: int = input_df.count()
    missing_exprs = []

    for field in input_df.schema.fields:
        column_name: str = field.name
        column_type = field.dataType

        if isinstance(column_type, StringType):
            clean_col = lower(trim(col(column_name)))

            missing_condition = (
                col(column_name).isNull()
                | (clean_col == "")
                | (clean_col.isin("none", "null", "nan", "na", "n/a"))
            )
        else:
            missing_condition = col(column_name).isNull()

        missing_exprs.append(
            count(when(missing_condition, column_name)).alias(column_name)
        )

    missing_row_df: DataFrame = input_df.select(missing_exprs)

    missing_values: dict = missing_row_df.collect()[0].asDict()

    rows: list[dict[str, object]] = []

    for column_name, missing_count in missing_values.items():
        missing_count_int = int(missing_count)

        rows.append(
            {
                "Column": column_name,
                "Number of Missing Values": missing_count_int,
                "Missing Percentage": round(
                    (missing_count_int / total_rows) * 100,
                    2
                ) if total_rows > 0 else 0.0,
            }
        )

    missing_df_out: DataFrame = spark.createDataFrame(rows)

    if show:
        display(
            missing_df_out.orderBy(
                col("Number of Missing Values").desc()
            )
        )

    return missing_df_out