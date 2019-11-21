# %%
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import string
from pyspark import sql
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec, Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, RandomForestClassificationModel, NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# %%
JSON_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler.json.gz'
TOKENIZED_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_tokenized.json'
DF_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_processed.csv'
DEBUG_CHECKPOINT_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/SparkCheckpoint'

# %%
sparkConf = SparkConf().setAppName('CMPE256')
sparkConf.set('spark.executor.memory', '16g')
sparkConf.set('spark.executor.cores', '5')
sparkConf.set('spark.cores.max', '40')
sparkConf.set('spark.driver.memory', '12g')
sparkConf.set('spark.driver.maxResultSize', '4g')

spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
spark.sparkContext.setCheckpointDir(DEBUG_CHECKPOINT_PATH)

# %%
dataSetRaw = spark.read.json(JSON_PATH)
dataSetRaw.show(25)

# %%
dataSet = dataSetRaw.withColumn('reviewNew', F.explode('review_sentences'))
columns = ['book_id', 'user_id', 'review_id', 'has_spoiler', 'rating'] + [dataSet.reviewNew[i] for i in range(2)]
dataSet = dataSet.select(columns).withColumnRenamed('reviewNew[0]', 'class').withColumnRenamed('reviewNew[1]', 'reviews')
dataSet.show(15)

# %%
import gensim.parsing.preprocessing as gsp
from pyspark.sql.functions import udf
from gensim import utils

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

@udf(returnType=StringType())
def cleanText(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

# %%
dataSet = dataSet.withColumn('cleanReview', cleanText(F.col('reviews'))).filter(F.col('cleanReview') != '')
dataSet.show()

# %%
trainDF, testDF = dataSet.randomSplit([0.8, 0.2])
# trainDF.show()
# testDF.show()

# %%
tokenizer = Tokenizer(inputCol="cleanReview", outputCol="tokens")
word2vec = Word2Vec(vectorSize=200, minCount=10, numPartitions=10, inputCol=tokenizer.getOutputCol(), outputCol="features")
pipeline = Pipeline(stages=[tokenizer, word2vec])
pipelineModel = pipeline.fit(trainDF)

# %%
pTrainDF = pipelineModel.transform(trainDF)
pTestDF = pipelineModel.transform(testDF)

# %%
pTrainDF = pTrainDF.withColumn('class', pTrainDF['class'].cast(IntegerType()))
pTestDF = pTestDF.withColumn('class', pTestDF['class'].cast(IntegerType()))

# %%
rForest = RandomForestClassifier(labelCol='class', featuresCol='features')
rForestModel = rForest.fit(pTestDF)

# %%
predictions = rForestModel.transform(pTestDF)

# %%
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction", metricName="f1")
evaluator.evaluate(predictions)

# %%

lr = LogisticRegression(featuresCol='features', labelCol='class')
lrModel = lr.fit(pTrainDF)
predictionsLR = lrModel.transform(pTestDF)
evaluator.evaluate(predictionsLR)

# %%
naiveBayes = NaiveBayes(featuresCol='features', labelCol='class')
naiveModel = naiveBayes.fit(pTrainDF)
predictionsNaive = naiveModel.transform(pTestDF)
evaluator.evaluate(predictionsNaive)


# %%
pipelineModel.save('D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/pipelineW2V')
rForestModel.save('D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/rForest')

#%%
pipelineModel = PipelineModel.load('D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/pipelineW2V')
rForestModel = RandomForestClassificationModel.load('D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/rForest')


# %%
