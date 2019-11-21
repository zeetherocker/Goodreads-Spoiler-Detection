# %%
import pandas as pd
import gensim.parsing.preprocessing as gsp
from gensim import utils
from pyspark import sql
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# %%
JSON_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler.json.gz'
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
filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text,
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
dataSet = dataSet.withColumn('class', dataSet['class'].cast(IntegerType()))
dataSet = dataSet.select('class', 'cleanReview').withColumnRenamed('cleanReview', 'reviews')

# %%
trainDF, testDF = dataSet.randomSplit([0.8, 0.2])
trainDF.show()
testDF.show()

# %%
tokenizer = Tokenizer(inputCol="reviews", outputCol="tokens")
countVector = CountVectorizer(inputCol=tokenizer.getOutputCol(), outputCol='features')
idf = IDF(inputCol=countVector.getOutputCol(), outputCol='idf')
pipeline = Pipeline(stages=[tokenizer, countVector, idf])
pipelineModel = pipeline.fit(trainDF)

# %%
pTrainDF = pipelineModel.transform(trainDF)
pTestDF = pipelineModel.transform(testDF)

# %%
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction", metricName="f1")
lr = LogisticRegression(featuresCol=idf.getOutputCol(), labelCol='class')
lrModel = lr.fit(pTrainDF)
predictionsLR = lrModel.transform(pTestDF)
evaluator.evaluate(predictionsLR)


# %%

inputComment = "Game of thrones is an awesome book. \
    George RR Martin has done a fantastic job at it.\n \
    Arya stark killed night king. \
    At the end of the third book we find out that Jon Snow is actually the son of rhaegar targaryen and lyanna stark. \
    Jon snow is not at all a bastard. \
    Bran stark becomes the king. \
    Arya stark. \
    Jon Snow was always the heir to the throne. \
    Arya Stark survives the whole battle of winterfell and goes on to kill the night king. \
    Game of the thrones has been legendary ever since it was released in 2001 it has never failed to move me and it has now gained serious viewership.\n \
    I am glad that dany died in the end specially after she went bad towards the end."

newLines = inputComment.split('\n')
lines = []
for line in newLines:
    lines += line.strip().split('. ')
    
demoDF = spark.createDataFrame(lines, StringType()).withColumnRenamed('value', 'reviews')
demoDF = demoDF.withColumn('cleanReview', cleanText(F.col('reviews'))).filter(F.col('cleanReview') != '')
demoDF = pipelineModel.transform(demoDF)
demoPred = lrModel.transform(demoDF)

l = demoPred.select(['review', 'prediction']).collect()
print("The predictions are: ")
for row in l:
    print(row.prediction, row.review)