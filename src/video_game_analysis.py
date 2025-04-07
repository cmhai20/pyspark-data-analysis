"""
Video Game Sales Analysis Module

This module analyzes video game sales data from 2006-2015,
identifying best-performing publishers and their sales trends.
"""

from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

def define_schema():
    """Define the schema for the video games sales data."""
    return StructType([
        StructField("title", StringType(), True),
        StructField("publisher", StringType(), True),
        StructField("developer", StringType(), True),
        StructField("release_date", DateType(), True),
        StructField("platform", StringType(), True),
        StructField("total_sales", DoubleType(), True),
        StructField("na_sales", DoubleType(), True),
        StructField("japan_sales", DoubleType(), True),
        StructField("pal_sales", DoubleType(), True),
        StructField("other_sales", DoubleType(), True),
        StructField("user_score", DoubleType(), True),
        StructField("critic_score", DoubleType(), True)
    ])

def load_sales_data(spark, file_path):
    """Load video game sales data from CSV."""
    schema = define_schema()
    required_columns = ["publisher", "release_date", "na_sales", "total_sales"]
    
    return spark.read \
        .option("header", "true") \
        .option("sep", "|") \
        .schema(schema) \
        .csv(file_path) \
        .select(*required_columns)

def analyze_publisher_sales(spark, file_path):
    """Analyze publisher sales for 2006-2015 period."""
    salesDF = load_sales_data(spark, file_path)
    
    # Extract the year and filter for 2006-2015
    sales0615DF = salesDF.withColumn("year", f.year("release_date")) \
                        .filter((f.col("year") >= 2006) & (f.col("year") <= 2015))
    
    # Find the publisher with the highest North America sales
    bestPublisherDF = sales0615DF.groupBy("publisher") \
        .agg(f.sum("na_sales").alias("total_na_sales")) \
        .orderBy(f.col("total_na_sales").desc()) \
        .limit(1)
    
    bestNAPublisher = bestPublisherDF.select("publisher").first()["publisher"]
    
    # Filter to games by the best North America publisher
    bestNADF = sales0615DF.filter(f.col("publisher") == bestNAPublisher)
    
    # Count titles with missing North America sales data
    titlesWithMissingSalesData = bestNADF.filter(f.col("na_sales").isNull()).count()
    
    # Aggregate yearly sales for North America and globally
    bestNAPublisherSales = bestNADF.groupBy("year").agg(
        f.round(f.sum("na_sales"), 2).alias("na_total"),
        f.round(f.sum("total_sales"), 2).alias("global_total")
    ).orderBy("year")
    
    print(f"The publisher with the highest total video game sales in North America is: '{bestNAPublisher}'")
    print(f"The number of titles with missing sales data for North America: {titlesWithMissingSalesData}")
    print("Sales data for the publisher:")
    bestNAPublisherSales.show()
    
    return bestNAPublisher, bestNAPublisherSales

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Video Game Sales Analysis") \
        .getOrCreate()
    
    # Set file path
    file_csv = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv"
    
    # Run analysis
    analyze_publisher_sales(spark, file_csv)
    
    # Stop Spark session
    spark.stop()
