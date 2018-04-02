from pyspark import SQLContext
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType
import re


def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer").getOrCreate()
    # sqlContext = SQLContext(sc)

    cases = sc.read.json("../data/sentencingCases.jsonl")
    df = cleanRdd(cases)
    # df.show()

    hashingTF = HashingTF(inputCol="filteredFullText", outputCol="rawFeatures", numFeatures=262144)
    featurizedData = hashingTF.transform(df)
    # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("title", "features").show()


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
              'para', 'am', 'me', 'law', 'justice', 'consider', 'make']

def cleanFullText(text):
    # remove whitespace and punctuation
    text = re.split('\s+', re.sub(r'[^\w\s]', '', text.lower()))
    return text

def cleanKeywords(text):
    # remove whitespace and punctuation
    text = re.split(' — ', text.lower())
    return text

def cleanRdd(df):
    df = df.withColumn("decisionDate", (f.col("decisionDate").cast("date")))

    cleanFT_udf = f.udf(cleanFullText, ArrayType(StringType()))
    df = df.withColumn("fullText", cleanFT_udf(df.fullText))

    cleanK_udf = f.udf(cleanKeywords, ArrayType(StringType()))
    df = df.withColumn("keywords", cleanK_udf(df.keywords))

    remover = StopWordsRemover(inputCol="fullText", outputCol="filteredFullText", stopWords=stop_words)
    df = remover.transform(df)

    return df


def text2int (textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring

main()



