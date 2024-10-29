from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.types import StructType, StructField, IntegerType
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
import pickle
import os
import sys
from flask import Flask,jsonify
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def aprior_Fun(df,ID):
    n=0
    result=[]
    apriori_model.sort_values(by=['Lift'])
    for _, row in apriori_model.iterrows():
        
        if n>=3:
            break
        elif ID in row['Antecedent'] :
            result.append(list(row['Consequent'])[0])
            n+=1    
    return f"{result}"

def ALS_Fun(model , ID):

    schema = StructType([StructField("userID", IntegerType(), True)])
    user_id = spark.createDataFrame([(ID,)], schema)    
    
    print(user_id)
    user_id.printSchema()
    user_id.show()
    # Number of items to recommend
    num_items = 3

    # Generate recommendations for the user
    recommendations = model.recommendForUserSubset(user_id, num_items)
    recommendations_exploded = recommendations.select("userID", explode("recommendations").alias("recommendation"))
    recommendations_with_ratings = recommendations_exploded.select(
        "userID",
        "recommendation.placeID",
        "recommendation.rating"
    )
    recommendations_with_ratings.show()

    # Extract recommended item IDs
    try:
        recommended_items = [row['placeID'] for row in recommendations.collect()[0]['recommendations']]
        user_id=user_id.toPandas()
        return f"{recommended_items}"
    except:
        return f"[1,2,3]"

def opinion_mining(lr,tfidf,comment):

    stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)    
    text = re.sub('[^a-zA-Z]', ' ', comment)
    text = text.lower()
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text 
            if not word in stuff_to_be_removed] 
    text1 = " ".join(text)
    data_cleaned = [text1]

    vector=tfidf.transform(data_cleaned)
    return 'Positive' if lr.predict(vector)[0]==1 else 'Negative'

# Initialize Spark session and load ALS model
spark = SparkSession.builder.master("local[*]").appName("RecommendationEngine").getOrCreate()
ALS_model = ALSModel.load("als_model")

# Initialize and load Apriori model
apriori_model = pickle.load(open('aprioi.sav', 'rb'))

# Initialize and load opinion mining model & tfidf
lr1 = pickle.load(open('opinion mining.sav', 'rb'))
tfidf1 =pickle.load(open('tfidf.sav', 'rb'))

app= Flask(__name__)

@app.route("/ALS/<int:ID>", methods=["GET"])
def ALS(ID):
    return ALS_Fun(ALS_model,ID)

@app.route("/apriori/<string:ID>", methods = ["GET"])
def Apriori(ID):
    return aprior_Fun(apriori_model , ID)

@app.route("/opinion mining/<string:comment>",methods= ["GET"])
def sentiment_analysis(comment):
    return opinion_mining(lr1,tfidf1,comment)

if __name__=='__main__':
    app.run(debug=True)
