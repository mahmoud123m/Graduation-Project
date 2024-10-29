from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# from pyspark.ml.tuning import TrainValidationSplit,ParamGridBuilder
# import pandas as pd
import os
import sys
from pyspark.sql import SparkSession
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.master("local[1]").appName("place_rating").getOrCreate()

#load and show data
train =  spark.read.option("header",True) \
              .csv('user_ratings.csv')
train.show()
train.printSchema()

#cast from string to integer
train=train.withColumn("userID",train['userID'].cast('int'))
train=train.withColumn("placeID",train['placeID'].cast('int'))
train=train.withColumn("place_rating",train['place_rating'].cast('int'))

#split the data
(train_data, valid_data)=train.randomSplit([0.8, 0.2])
train.printSchema()

#prepare the als model
als= ALS(userCol="userID" ,itemCol="placeID", ratingCol="place_rating",
         maxIter=10,
          regParam=0.5,
          rank=100,
          coldStartStrategy="nan", 
          nonnegative = True)

#evaluator
evaluater= RegressionEvaluator(metricName="rmse",labelCol="place_rating",predictionCol="prediction")

#Fit the model
model = als.fit(train)

# bestmodel=model.bestModel
prediction=model.transform(valid_data)
prediction=prediction.na.drop("any")
prediction.show()

#evaluate
rmse=evaluater.evaluate(prediction)
print("RMSE: "+ str(rmse))

#######################################################   Save the model   ############################################# 
# model.save("als_model")

