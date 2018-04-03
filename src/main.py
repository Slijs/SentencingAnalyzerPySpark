from pyspark import SQLContext
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer, NGram
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType
import re

def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer").getOrCreate()
    sqlContext = SQLContext(sc)
    print(sc.version)

    cases = sc.read.json("../data/sentencingCases.jsonl")
    df = cleanDf(cases)
    # tokenizer = Tokenizer(inputCol="fullText", outputCol="words")
    # df = tokenizer.transform(cases)
    # remover = StopWordsRemover(inputCol="words", outputCol="filteredWords", stopWords=stop_words)
    # df = remover.transform(df)
    # df.select("filteredFullText").show(truncate=False)

    # compute n-grams
    ngram = NGram(n=3, inputCol="filteredFullText", outputCol="ngrams")
    ngramDataFrame = ngram.transform(df)
    ngramDataFrame.select("ngrams").show(truncate=False)

    # hashingTF = HashingTF(inputCol="filteredFullText", outputCol="rawFeatures", numFeatures=262144)
    # featurizedData = hashingTF.transform(df)
    # # alternatively, CountVectorizer can also be used to get term frequency vectors
    #
    # idf = IDF(inputCol="rawFeatures", outputCol="features")
    # idfModel = idf.fit(featurizedData)
    # rescaledData = idfModel.transform(featurizedData)
    #
    # rescaledData.select("title", "features").show()


stop_words = ['the', 'of', 'to', 'and', 'a', 'in', 'that', 'is', 'for', 'was', 'be', 'on', 'not', 'his', 'he', 'as',
              'with', 'by', 'or', 'at', 'an', 'this', 'has', 'i', 'from', 'it', 'have', 'which', 'had', 'her', 'are',
              'been', 'would', 'were', 'any', 'will', 'she', 'there', 'should', 'other', 'must', 'case',
              'order', 'but', 'under', 'him', 'may', 'if', 'did', 'he', 'who', 'into', 'where', 'they', 'these',
              'than', 'out', 'such', 'canlii', 'only', 'in', 'made', 'no', 'more', 'can', 'my',
              'their', 'do', 'you', 'also', 'some', 'what', 'being', 'does', 'because', 'whether', 'both', 'could',
              'about', 'those', 'said', 'its', 'so', 'set', 'very', 'respect', 'cases', 'against', 'ms', 'fact',
              'para', 'am', 'me', 'law', 'justice', 'consider', 'make', '', ',']

def cleanFullText(text):
    # remove whitespace and punctuation
    text = re.split(r'\s+', re.sub(r'[^\w\s]', ' ', text.lower()))

    for word in text:
        word = text2int(word)
    return text

def cleanKeywords(text):
    # remove whitespace and punctuation
    text = re.split(' — ', text.lower())
    return text

def cleanDf(df):
    df = df.withColumn("decisionDate", (f.col("decisionDate").cast("date")))

    cleanFT_udf = f.udf(cleanFullText, ArrayType(StringType()))
    df = df.withColumn("fullTextCleaned", cleanFT_udf(df.fullText))

    cleanK_udf = f.udf(cleanKeywords, ArrayType(StringType()))
    df = df.withColumn("keywords", cleanK_udf(df.keywords))

    remover = StopWordsRemover(inputCol="fullTextCleaned", outputCol="filteredFullText", stopWords=stop_words)
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



