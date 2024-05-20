from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Car Price Prediction") \
    .getOrCreate()

# Load the data
df = spark.read.csv('/kaggle/input/second-hand-car-price-prediction/cars.csv', header=True, inferSchema=True)

# Set up StringIndexer to handle categorical variables
indexers = [
    StringIndexer(inputCol='Brand', outputCol='Brand_Indexed'),
    StringIndexer(inputCol='Model', outputCol='Model_Indexed'),
    StringIndexer(inputCol='Fuel_Type', outputCol='Fuel_Type_Indexed'),
    StringIndexer(inputCol='Transmission', outputCol='Transmission_Indexed'),
    StringIndexer(inputCol='Owner_Type', outputCol='Owner_Type_Indexed')
]

# Build the pipeline
pipeline = Pipeline(stages=indexers)
df_transformed = pipeline.fit(df).transform(df)

# Calculate the car age and drop the year column
df_transformed = df_transformed.withColumn('Car_Age', lit(2024) - col('Year')).drop('Year')

# Drop na values
df_transformed = df_transformed.na.drop()

# Select and save the necessary columns
output_columns = ['Transmission_Indexed', 'Mileage', 'Owner_Type_Indexed', 'Engine', 'Power', 'Car_Age', 'Price']
df_transformed.select(output_columns).write.csv('output.csv', header=True)

# Stop the Spark session
spark.stop()