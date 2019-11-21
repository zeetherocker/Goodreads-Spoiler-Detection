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
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# %%
JSON_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler.json.gz'
TOKENIZED_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_tokenized.json'
DF_CSV_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Dataset/goodreads_reviews_spoiler_processed.csv'
DEBUG_CHECKPOINT_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/SparkCheckpoint'
IMG_PATH = 'D:/College_Stuff/3rd_Sem/CMPE256/Project/Images/'

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
dataSet.show()

# %%
columns = ['book_id', 'user_id', 'review_id', 'has_spoiler', 'rating'] + [dataSet.reviewNew[i] for i in range(2)]
dataSet = dataSet.select(columns).withColumnRenamed('reviewNew[0]', 'class').withColumnRenamed('reviewNew[1]', 'reviews')
dataSet.show(15)

# %%
processedDF = dataSet.groupBy('review_id', 'class').agg(F.concat_ws(' ', F.collect_list('reviews')).alias('review'))
processedDF.show()

# %%
dataSet = processedDF.join(dataSetRaw, on='review_id', how='left')
dataSet.show()

# %%
count = dataSet.groupBy('class').count().collect()
count

# %%
neutralCount, spoilerCount = count
neutralCount = neutralCount['count']
spoilerCount = spoilerCount['count']
totalCount = neutralCount + spoilerCount

print('The dataset contains {} review sentences with {} spoiler reviews and {} non-spoiler reviews'\
            .format(totalCount, spoilerCount, neutralCount))

plt.bar(['Spoiler', 'Non-Spoiler'], [spoilerCount, neutralCount], width = 0.5)
plt.ylabel('No. of reviews')
plt.savefig(IMG_PATH + 'numbers.png', format='png', transparent=True)
plt.show()

# %%
punctuationUDF = F.udf(lambda sentence: sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip(), StringType())
dataSet = dataSet.withColumn('review', punctuationUDF(F.col('review')))

# %%
temp = dataSet.withColumn('length', F.length('review'))
reviewDF = temp.groupBy('review_id', 'has_spoiler').agg(F.sum('length').alias('tLength'))
avgSpoilerLength = reviewDF.groupBy('has_spoiler').agg(F.mean('tLength').alias('avgLength'))
avgSpoilerLen, avgNeutralLen = avgSpoilerLength.collect()

plt.bar(['Spoiler', 'Non-Spoiler'], [2022, 1000], width = 0.5)
plt.ylabel('Average Review Length')
plt.savefig(IMG_PATH + 'avgLength.png', format='png', transparent=True)
plt.show()

# %%
avgLengthDF = temp.groupBy('class').agg(F.mean('length').alias('avgLength'))
avgNeutralLen, avgSpoilerLen = avgLengthDF.collect()

plt.bar(['Spoiler', 'Non-Spoiler'], [avgSpoilerLen['avgLength'], avgNeutralLen['avgLength']], width = 0.5)
plt.ylabel('Average Review Length (Sentence-wise)')
plt.savefig(IMG_PATH + 'avgLength_Sentence.png', format='png', transparent=True)
plt.show()

# %%
reviewLengths = temp.select('class', 'length').collect()

# %%

y = [int(row['length']) for row in reviewLengths if (int(row['class']) == 0)]
mx, mn = max(y), min(y)
ptp = mx - mn
y1 = [(i-mn)/ptp for i in y]

y = [int(row['length']) for row in reviewLengths if (int(row['class']) == 1)]
mx, mn = max(y), min(y)
ptp = mx - mn
y2 = [(i-mn)/ptp for i in y]

# %%
x = np.arange(0.0, 1.001, 0.001)
xLen = len(x)
y1Val, y2Val = [], []
inc1 = len(y1) // xLen
inc2 = len(y2) // xLen
i = 0
while (i < len(y1)) and (len(y1Val) < xLen):
    y1Val.append(y1[i])
    i += inc1
i = 0
while (i < len(y2))and (len(y2Val) < xLen):
    y2Val.append(y2[i])
    i += inc2

(len(y1Val), len(y2Val), xLen)

# %%
plt.plot(x, y1Val, 'go--', linewidth=2, markersize=0, label='Non-Spoiler')
plt.plot(x, y2Val, 'ro-', linewidth=2, markersize=0, label='Spoiler')
plt.ylabel('Average Sentence Length')
plt.legend(loc='upper left')
plt.title('Review Sentence Length')
plt.savefig(IMG_PATH + 'length.png', format='png', transparent=True)
plt.show()

y1Val.sort()
y2Val.sort()

plt.plot(x, y1Val, 'go--', linewidth=2, markersize=0, label='Non-Spoiler')
plt.plot(x, y2Val, 'ro-', linewidth=2, markersize=0, label='Spoiler')
plt.ylabel('Average Sentence Length')
plt.legend(loc='upper left')
plt.title('Review Sentence Length')
plt.savefig(IMG_PATH + 'lengthSorted.png', format='png', transparent=True)
plt.show()



# %%
stopWords = list(set(nltk.corpus.stopwords.words('english'))) + ['']
tokenizer = Tokenizer(inputCol='review', outputCol='tokens')
stopWordRemover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='stoppedWords').setStopWords(stopWords)
pipeline = Pipeline(stages=[tokenizer, stopWordRemover])
dataSet = pipeline.fit(dataSet).transform(dataSet)

# %%
newLengthDF = dataSet.withColumn('newLength', F.size(stopWordRemover.getOutputCol()))

# %%
newSentenceLen = newLengthDF.select('class', 'newLength').collect()

#%%
y = [int(row['newLength']) for row in reviewLengths if (int(row['class']) == 0)]
mx, mn = max(y), min(y)
ptp = mx - mn
y1 = [(i-mn)/ptp for i in y]

y = [int(row['newLength']) for row in reviewLengths if (int(row['class']) == 1)]
mx, mn = max(y), min(y)
ptp = mx - mn
y2 = [(i-mn)/ptp for i in y]

# %%
x = np.arange(0.0, 1.001, 0.001)
xLen = len(x)
y1Val, y2Val = [], []
inc1 = len(y1) // xLen
inc2 = len(y2) // xLen
i = 0
while (i < len(y1)) and (len(y1Val) < xLen):
    y1Val.append(y1[i])
    i += inc1
i = 0
while (i < len(y2))and (len(y2Val) < xLen):
    y2Val.append(y2[i])
    i += inc2

(len(y1Val), len(y2Val), xLen)

# %%
plt.plot(x, y1Val, 'go--', linewidth=2, markersize=0, label='Non-Spoiler')
plt.plot(x, y2Val, 'ro-', linewidth=2, markersize=0, label='Spoiler')
plt.ylabel('Average Sentence Length')
plt.legend(loc='upper left')
plt.title('Review Length after preprocessing')
plt.savefig(IMG_PATH + 'stoppedLen.png', format='png', transparent=True)
plt.show()

y1Val.sort()
y2Val.sort()

plt.plot(x, y1Val, 'go--', linewidth=2, markersize=0, label='Non-Spoiler')
plt.plot(x, y2Val, 'ro-', linewidth=2, markersize=0, label='Spoiler')
plt.ylabel('Average Review Length')
plt.legend(loc='upper left')
plt.title('Review Length after preprocessing')
plt.savefig(IMG_PATH + 'stoppedLenSorted.png', format='png', transparent=True)
plt.show()

# %%
idfCalc = dataSet.select('review_id', 'user_id', 'book_id', 'stoppedWords', 'class').withColumn('doc_id', F.monotonically_increasing_id())
idfCalc = idfCalc.withColumn('tokens', F.explode('stoppedWords'))

# %%
from pyspark.sql.window import Window

w = Window.partitionBy(idfCalc['doc_id'])

book_tf = idfCalc.groupBy('doc_id', 'tokens', 'book_id', 'class')\
    .agg(
         (F.count('*')/F.sum(F.count('*')).over(w)).alias('tf')
        )\
    .orderBy('doc_id', ascending=True)

del w

# %%
w = Window.partitionBy(idfCalc['tokens'])

c_d = idfCalc.select('doc_id').distinct().count()

book_idf = book_tf.groupBy('tokens', 'doc_id', 'book_id', 'class', 'tf').agg(
        F.log(F.lit(c_d)/F.count('*').over(w)).alias('idf')
    )\
    .orderBy('doc_id', ascending=True)

del w

# %%
book_tfidf = book_idf.withColumn('tf_idf', F.col('tf') * F.col('idf'))

# %%
temp = book_tfidf.select('class', 'tokens', 'tf_idf', 'book_id')

# %%

# %%
avgSpecifivity = temp.groupBy('class').agg(F.max('tf_idf')).collect()
avgSpecNeutral, avgSpecSpoiler = avgSpecifivity

# %%

plt.bar(['Spoiler', 'Non-Spoiler'], [avgSpecSpoiler, avgSpecNeutral], width = 0.5)
plt.title('Item specificity')
plt.ylabel('tf-idf')
plt.savefig(IMG_PATH + 'itemSpecificity.png', format='png', transparent=False)
plt.show()

# %%
tokenList = temp.filter('book_id == 13422727').select('tokens', 'tf_idf').collect()

# %%
tokenList

# %%
tempTokenList = [(row['tokens'], row['tf_idf']) for row in tokenList]
tempTokenList.sort(key=lambda x: x[1], reverse=True)

# %%
filteredWords = [token for token, _ in tempTokenList[:100]] 

# %%
res = ['dany'] * 5

newFilter = filteredWords + res

# %%
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(newFilter))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(IMG_PATH + 'wordCloud.jpg', quality=95, dpi=600)
plt.show()

# %%
userInfo = dataSetRaw.groupBy('user_id').agg(F.sum(F.when(F.col('has_spoiler'), 1).otherwise(0)).alias('spoilerCnt'), F.count('*').alias('totalCnt'))
userInfo = userInfo.withColumn('ratio', F.col('spoilerCnt')/F.col('totalCnt'))

userInfoList = userInfo.select('ratio').collect()

# %%
bookInfo = dataSetRaw.groupBy('book_id').agg(F.sum(F.when(F.col('has_spoiler'), 1).otherwise(0)).alias('spoilerCnt'), F.count('*').alias('totalCnt'))
bookInfo = bookInfo.withColumn('ratio', F.col('spoilerCnt')/F.col('totalCnt'))

bookInfoList = bookInfo.select('ratio').collect()

# %%
userInfoList = [row['ratio'] for row in userInfoList]
bookInfoList = [row['ratio'] for row in bookInfoList]

# %%

y = userInfoList
mx, mn = max(y), min(y)
ptp = mx - mn
y1 = [(i-mn)/ptp for i in y]

y = bookInfoList
mx, mn = max(y), min(y)
ptp = mx - mn
y2 = [(i-mn)/ptp for i in y]

# %%
x = np.arange(0.0, 1.01, 0.01)
xLen = len(x)
y1Val, y2Val = [], []
inc1 = len(y1) // xLen
inc2 = len(y2) // xLen
i = 0
while (i < len(y1)) and (len(y1Val) < xLen):
    y1Val.append(y1[i])
    i += inc1
i = 0
while (i < len(y2))and (len(y2Val) < xLen):
    y2Val.append(y2[i])
    i += inc2

(len(y1Val), len(y2Val), xLen)


#%%
y1Val.sort()
y2Val.sort()


# %%
plt.plot(x, y1Val, 'go--', linewidth=2, markersize=0, label='User-ratio')
plt.plot(x, y2Val, 'ro-', linewidth=2, markersize=0, label='Book-ratio')
plt.ylabel('Spoiler %age')
plt.legend(loc='upper left')
plt.title('User/Book Spoiler ratio')
plt.savefig(IMG_PATH + 'user_book_ratio.png', format='png')
plt.show()