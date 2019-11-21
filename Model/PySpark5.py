# %%
import nltk
import os
import pandas as pd
import pyspark.sql.functions as F
import string
from pyspark import sql
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.types import *
from pyspark.ml.feature import Word2Vec, Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel, NaiveBayes, NaiveBayesModel, LinearSVC, LinearSVCModel
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# %%
JSON_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler.json'
TOKENIZED_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_tokenized.json'
DF_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_processed.csv'
DEBUG_CHECKPOINT_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/SparkCheckpoint'
LR_MODEL_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/lr_model'
PIPELINE_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Models/pipeline'


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
dataSet = spark.read.json(JSON_PATH)
dataSet.show(25)

# %%
dataSet = dataSet.withColumn('reviewNew', F.explode('review_sentences'))
dataSet.show()

# %%
columns = ['review_id'] + [dataSet.reviewNew[i] for i in range(2)]
dataSet = dataSet.select(columns).withColumnRenamed('reviewNew[0]', 'class').withColumnRenamed('reviewNew[1]', 'reviews')
dataSet.show(15)

# %%
processedDF = dataSet.groupBy('review_id', 'class').agg(F.concat_ws(' ', F.collect_list('reviews')).alias('review')).select(['class', 'review'])
processedDF.show()

# %%
punctuationUDF = F.udf(lambda sentence: sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip(), StringType())
processedDF = processedDF.withColumn('review', punctuationUDF(F.col('review')))
processedDF.show()

# %%
processedDF = processedDF.withColumn('class', processedDF['class'].cast(IntegerType()))

# %%
if False and os.path.exists(DF_CSV_PATH):
    df = pd.read_csv(DF_CSV_PATH)
else:
    df = processedDF.toPandas()
    df.to_csv(DF_CSV_PATH, index=False)

df.head()

#%%
df = df.astype({'class': 'int32'})

# %%
trainDF = df.sample(frac=0.8, random_state=80)
testDF = df.drop(trainDF.index)

#%%
trainDF['class'].isnull().sum()

# %%
trainDF, testDF = sql.SQLContext(spark.sparkContext).createDataFrame(trainDF), sql.SQLContext(spark.sparkContext).createDataFrame(testDF)
# trainDF.show()
# testDF.show()

#%%
print(trainDF.count())
print(trainDF.filter(F.col('class') == 1).count())
print(trainDF.filter(F.col('class') == 0).count())

# %%
stopWords = list(set(nltk.corpus.stopwords.words('english'))) + ['']

tokenizer = Tokenizer(inputCol='review', outputCol='tokens')
stopWordRemover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='stoppedWords').setStopWords(stopWords)
countVector = CountVectorizer(inputCol=stopWordRemover.getOutputCol(), outputCol='vectors')
idf = IDF(inputCol=countVector.getOutputCol(), outputCol='idf')
pipline = Pipeline(stages=[tokenizer, stopWordRemover, countVector, idf])
model = pipline.fit(trainDF)
ptrainDF = model.transform(trainDF)
ptestDF = model.transform(testDF)
ptrainDF.show()

# %%
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction", metricName="f1")

# %%
lr = LogisticRegression(featuresCol=idf.getOutputCol(), labelCol='class')
lrModel = lr.fit(ptrainDF)
predictionsLR = lrModel.transform(ptestDF)
evaluator.evaluate(predictionsLR)

# %%
lda = LDA(featuresCol=idf.getOutputCol(), maxIter=10, k=2)
ldaModel = lda.fit(ptrainDF)

# %%
naiveBayes = NaiveBayes(featuresCol=idf.getOutputCol(), labelCol='class')
naiveModel = naiveBayes.fit(ptrainDF)
predictionsNaive = naiveModel.transform(ptestDF)
evaluator.evaluate(predictionsNaive)

# %%
svc = LinearSVC(featuresCol=idf.getOutputCol(), labelCol='class')
svcModel = svc.fit(ptrainDF)
predictionsSVC = svcModel.transform(ptestDF)
evaluator.evaluate(predictionsSVC)

# %%
model.save(PIPELINE_PATH)
lrModel.save(LR_MODEL_PATH)

# %%
model = PipelineModel.load(PIPELINE_PATH)
lrModel = LogisticRegressionModel.load(LR_MODEL_PATH)

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

demoDF = spark.createDataFrame(lines, StringType()).withColumnRenamed('value', 'review')

punctuationUDF = F.udf(lambda sentence: sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip(), StringType())
demoDF = demoDF.withColumn('review', punctuationUDF(F.col('review')))
demoDF = model.transform(demoDF)
demoPred = lrModel.transform(demoDF)

l = demoPred.select(['review', 'prediction']).collect()
print("The predictions are: ")
for row in l:
    print(row.prediction, row.review)

# %%
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictionsLR.withColumnRenamed('class', 'label'), {evaluator.metricName: "f1"})))