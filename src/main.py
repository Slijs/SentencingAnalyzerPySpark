from pyspark import SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
# from pyspark.sql.functions import col, lit
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType
import re


# Constants
# myMaxIter = 5
# myRank = 70
# myRegParam = 0.01
# myColdStartStrategy = 'drop'


def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer").getOrCreate()
    # sqlContext = SQLContext(sc)

    cases = sc.read.json("../data/sentencingCases.jsonl")
    df = cleanRdd(cases)
    df.show()
    # df.printSchema()
    # cases = cases.withColumn("decisionDate", (f.col("decisionDate").cast("date")))

    # SHOW DATABASE COUNT
    # cases = spark.read.option("multiLine", True).option("mode", "PERMISSIVE").json("../data/sentencingCases.json")
    # dbs = cases.withColumn("numInDB", lit(1))
    # dbs.show()
    # dbs.groupBy("databaseID").sum("numInDB").withColumnRenamed("sum(numInDB)", "numInDB").sort("numInDB", ascending=False).show(40)

    # ITERATE OVER ITEMS IN DATAFRAME
    # iterable = dbCounts.sort("numInDB").collect()
    # for db in iterable:
    #     print(db[0] + ": " + db[1])

    # SHOW KEYWORD COUNT
    # cases.withColumn('keyword', f.explode(f.split(f.col('keywords'), ' — ')))\
    #     .groupBy('keyword')\
    #     .count()\
    #     .sort('count', ascending=False)\
    #     .show()

    # SHOW WORD COUNT FROM FULLTEXT
    # stop_words = ['the', 'of', 'to', 'and', 'a', 'in', 'that', 'is', 'for', 'was', 'be', 'on', 'not', 'his', 'he', 'as',
    #               'with', 'by', 'or', 'at', 'an', 'this', 'has', 'I', 'from', 'it', 'have', 'which', 'had', 'her', 'are',
    #               'The', 'been', 'would', 'were', 'any', 'will', 'she', 'there', 'should', 'other', 'must', 'case',
    #               'order', 'but', 'under', 'him', 'may', 'if', 'did', 'He', 'who', 'into', 'where', 'they', 'these',
    #               'than', 'out', 'such', '(CanLII),', 'CanLII', 'only', 'In', 'made', 'No.', 'more', 'can', 'my',
    #               'their', 'do', 'you', 'also', 'some', 'what', 'being', 'does', 'because', 'whether', 'both', 'could',
    #               'about', 'those', 'said', 'its', 'so', 'set', 'very', 'respect', 'cases', 'against', 'Ms.', 'fact',
    #               'para.', 'It', 'am', 'me', 'law', 'Justice', 'consider', 'make', 'of\nthe']
    # cases.withColumn('word', f.explode(f.col('fullText'))) \
    #     .groupBy('word') \
    #     .count() \
    #     .sort('count', ascending=False) \
    #     .show(80)

    # # FP-Growth
    # casesWithWords = cases.withColumn('words', f.split(f.col('fullText'), ' '))
    # # casesWithWords.printSchema()
    # # casesWithWords.show()
    #
    # fpGrowth = FPGrowth(itemsCol="words", minSupport=0.5, minConfidence=0.6)
    # model = fpGrowth.fit(casesWithWords)
    #
    # # Display frequent itemsets.
    # model.freqItemsets.show()
    #
    # # Display generated association rules.
    # model.associationRules.show()
    #
    # # transform examines the input items against all the association rules and summarize the
    # # consequents as prediction
    # model.transform(casesWithWords).show()

# def isStopWord(word):
#     stop_words = ['the', 'of', 'to', 'and', 'a', 'in', 'that', 'is', 'for', 'was', 'be', 'on', 'not', 'his', 'he', 'as',
#                  'with', 'by', 'or', 'at']
#     for w in stop_words:
#         if w == word:
#             return True
#     return False

stop_words = ['the', 'of', 'to', 'and', 'a', 'in', 'that', 'is', 'for', 'was', 'be', 'on', 'not', 'his', 'he', 'as',
              'with', 'by', 'or', 'at', 'an', 'this', 'has', 'i', 'from', 'it', 'have', 'which', 'had', 'her', 'are',
              'been', 'would', 'were', 'any', 'will', 'she', 'there', 'should', 'other', 'must', 'case',
              'order', 'but', 'under', 'him', 'may', 'if', 'did', 'he', 'who', 'into', 'where', 'they', 'these',
              'than', 'out', 'such', 'canlii', 'only', 'in', 'made', 'no', 'more', 'can', 'my',
              'their', 'do', 'you', 'also', 'some', 'what', 'being', 'does', 'because', 'whether', 'both', 'could',
              'about', 'those', 'said', 'its', 'so', 'set', 'very', 'respect', 'cases', 'against', 'ms', 'fact',
              'para', 'it', 'am', 'me', 'law', 'justice', 'consider', 'make']

def cleanFullText(text):
    # remove whitespace and punctuation
    text = re.split('\s+', re.sub(r'[^\w\s]', '', text.lower()))
    return text

def cleanRdd(df):
    df = df.withColumn("decisionDate", (f.col("decisionDate").cast("date")))

    clean_udf = f.udf(cleanFullText, ArrayType(StringType()))
    df = df.withColumn("fullText", clean_udf(df.fullText))

    remover = StopWordsRemover(inputCol="fullText", outputCol="filteredFullText", stopWords=stop_words)
    df = remover.transform(df)

    return df

main()



