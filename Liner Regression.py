# Databricks notebook source
import pyspark

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Lr').getOrCreate()

# COMMAND ----------

data = spark.read.csv('/FileStore/tables/Ecommerce_Customers.csv',inferSchema=True,header=True)

# COMMAND ----------

data.show()

# COMMAND ----------

data.printSchema()
data.columns

# COMMAND ----------

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import  VectorAssembler

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['Avg Session Length','Time on App',
                                        'Time on Website','Length of Membership'],
                           outputCol = 'features')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

output.printSchema()

# COMMAND ----------

output.select('features').show(truncate=False)

# COMMAND ----------

final_data = output.select('features','Yearly Amount Spent')

# COMMAND ----------

final_data.show(truncate=False)

# COMMAND ----------

training_data,test_data = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

training_data.describe().show()

# COMMAND ----------

test_data.describe().show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

lr = LinearRegression(labelCol ='Yearly Amount Spent')

# COMMAND ----------

lr_model = lr.fit(training_data)

# COMMAND ----------

test_results = lr_model.evaluate(test_data) 


# COMMAND ----------

# Difference between actual value and predected value
test_results.residuals.show()

# COMMAND ----------

#root mean squre:
test_results.rootMeanSquaredError

# COMMAND ----------

# r2 
# model having 98% of variance in data
test_results.r2

# COMMAND ----------

final_data.describe().show()

# COMMAND ----------

unlabel_data = test_data.select('features')

# COMMAND ----------

unlabel_data.show()

# COMMAND ----------

predictions = lr_model.transform(unlabel_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------


