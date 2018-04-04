from pyspark import SQLContext
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer, NGram, CountVectorizer, MinHashLSH, Word2Vec
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType
import re





def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer").getOrCreate()
    sqlContext = SQLContext(sc)
    print(sc.version)

    # main df
    cases = sc.read.json("../data/sentencingCases.jsonl")
    df = cleanDf(cases)

    # create the search df
    df = extractOffenseKeywords(df)
    # df.select("Title", "caseID", "offenseKeywords").show(200, truncate=False)
    dfSearch = sc.createDataFrame(searchData, ["term", "offenseKeywords"])

    # tokenizer = Tokenizer(inputCol="fullText", outputCol="words")
    # df = tokenizer.transform(cases)
    # remover = StopWordsRemover(inputCol="words", outputCol="filteredWords", stopWords=stop_words)
    # df = remover.transform(df)
    # df.select("filteredFullText").show(truncate=False)

    # CLASSIFICATION OF OFFENSE
    # compute n-grams
    # ngram = NGram(n=2, inputCol="filteredFullText", outputCol="ngrams")
    # ngramDataFrame = ngram.transform(df)
    # ngramDataFrame.select("ngrams").show(truncate=False)
    #
    # word2vec = Word2Vec(inputCol="filteredFullText", outputCol="vectors")
    # model = word2vec.fit(df)
    # model.getVectors().show(truncate=False)
    # result = model.transform(df)
    # result.show()
    # for feature in result.select("vectors").take(3):
    #     print(feature)

    hashingTF = HashingTF(inputCol="offenseKeywords", outputCol="rawFeatures", numFeatures=1000)
    result = hashingTF.transform(df)
    resultSearch = hashingTF.transform(dfSearch)
    # # alternatively, CountVectorizer can also be used to get term frequency vectors



    # cv = CountVectorizer(inputCol="offenseKeywords", outputCol="rawFeatures", vocabSize=500)
    # model = cv.fit(df)
    # result = model.transform(df)
    # modelSearch = cv.fit(dfSearch)
    # resultSearch = modelSearch.transform(dfSearch)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(result)
    rescaledData = idfModel.transform(result)
    idfModelSearch = idf.fit(resultSearch)
    rescaledDataSearch = idfModelSearch.transform(resultSearch)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", seed=12345, numHashTables=10)
    modelMH = mh.fit(rescaledData)
    transformedData = modelMH.transform(rescaledData)

    modelMHSearch = mh.fit(rescaledDataSearch)
    transformedDataSearch = modelMH.transform(rescaledDataSearch)
    # transformedDataSearch.show()
    # transformedDataSearch.printSchema()
    #
    # asd = transformedDataSearch.rdd.collect()
    # print(asd[:][4])
    # # print(offenseKeywordHashes)

    # categorizedDf = transformedData.alias('a').join(transformedDataSearch.alias('b'), f.col('b.id') == f.col('a.id')).select([f.col('a.'+xx) for xx in transformedData.columns] + [f.col('b.other1'),f.col('b.other2')])
    # categorizedDf = searchForCategories(modelMHSearch, transformedDataSearch, transformedData)
    # categorizedDf.show()

    categorizedDf = modelMHSearch.approxSimilarityJoin(transformedDataSearch, transformedData, 0.88, distCol="JaccardDistance")
    categorizedDf.select([f.col('datasetA.term')] + [f.col('datasetB.caseID')] + [f.col("JaccardDistance")]) \
        .orderBy('caseID', 'JaccardDistance').show(200)

    # QUANTIZATION OF SENTENCE DURATION

    # SPLIT BY INDIGENOUS AND NON-INDIGENOUS

    # VISUALIZE RESULTS


stop_words = ['the', 'of', 'to', 'and', 'a', 'in', 'that', 'is', 'for', 'was', 'be', 'on', 'not', 'his', 'he', 'as',
              'with', 'by', 'or', 'at', 'an', 'this', 'has', 'i', 'from', 'it', 'have', 'which', 'had', 'her', 'are',
              'been', 'would', 'were', 'any', 'will', 'she', 'there', 'should', 'other', 'must', 'case',
              'order', 'but', 'under', 'him', 'may', 'if', 'did', 'he', 'who', 'into', 'where', 'they', 'these',
              'than', 'out', 'such', 'canlii', 'only', 'in', 'made', 'no', 'more', 'can', 'my',
              'their', 'do', 'you', 'also', 'some', 'what', 'being', 'does', 'because', 'whether', 'both', 'could',
              'about', 'those', 'said', 'its', 'so', 'set', 'very', 'respect', 'cases', 'against', 'ms', 'fact',
              'para', 'am', 'me', 'law', 'justice', 'consider', 'make', '', ',', 's', 'c', 'r', 'mr', 'wi']

searchData = [("assault", ['aggravated', 'assaulted', 'domestic', 'fight', 'assault', 'assaulting', 'threats', 'bodily', 'harm', 'attacked', 'attack', 'punched', 'punch']),
              ("sexual offences", ['sexual', 'sex', 'consent', 'abuse', 'abused', 'rape', 'incest', 'molester', 'touching', 'penis', 'vagina', 'breasts', 'breast', 'grope', 'groped']),
              ("homicide", ['manslaughter', 'murder', 'death', 'weapon', 'kill', 'killing', 'deceased', 'meditated', 'premeditated', 'died', 'accidental']),
              ("terrorism", ['group', 'terrorism', 'terrorist']),
              ("drug offences", ['drug', 'drugs', 'trafficking', 'crack', 'cocaine', 'heroin', 'kilogram', 'kilograms', 'pound', 'pounds', 'ounces', 'grams', 'marijuana', 'intoxicating', 'methamphetamine', 'meth', 'lsd']),
              ("robbery", ['robbery', 'robbed', 'rob', 'stole', 'stolen', 'break', 'enter', 'theft', '348']),
              ("weapon", ['weapon', 'firearm', 'firearms', 'pistol', 'rifle', 'knife', 'stabbed', 'stabbing', 'firing', 'fired', 'shooting', 'armed']),
              ("fraud", ['forgery', 'forging', 'fraud', 'impersonation', 'cheque', 'cheques', 'sum', 'financial', 'money', 'monies', 'deprivation', 'fraudulent', 'defraud', 'defrauded', 'defrauding', 'deceits', 'deceit', 'falsehood', 'breach', 'trust', 'con', 'artist', 'forgery']),
              ("child pornography", ['child', 'pornography', 'vile', 'disgusting', 'distribution', 'repulsive', 'underage']),
              ("mischief", ['mischief']),
              ("driving offences", ['253', 'driving', 'highway', 'traffic', 'suspended', 'hta', 'stunt', 'plates', 'careless', 'automobile', 'motor', 'vehicle', 'operate']),
              ("court-related offences", ['perjury', 'breaching', 'breach', 'condition', 'comply', '731', '139', '145', '264']),
              ("tax offences", ['evading', 'evade', 'tax', 'income', 'taxation', 'hiding'])]

def cleanFullText(text):
    # remove whitespace and punctuation
    text = re.split(r'\s+', re.sub(r'[^\w\s]', ' ', re.sub(r'\W*\b[^\W\d]{1,2}\b', ' ', text.lower())))

    for index, word in enumerate(text):
        text[index] = text2int(word).strip()
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


def offenseKeywords(textArray):
    words = []
    for word in textArray:
        for category in searchData:
            for keyword in category[1]:
                if word == keyword:
                    words.append(word)
                    break
    return words


def extractOffenseKeywords(df):
    offenseKeywords_udf = f.udf(offenseKeywords, ArrayType(StringType()))
    df = df.withColumn("offenseKeywords", offenseKeywords_udf(df.filteredFullText))
    return df


# def findOffenseCategory(hash):
#     model.approxNearestNeighbors(dfToSearch, hash, 2).show()
#     asdf = []
#     asdf.append('hey')
#     return asdf
#
#
# def searchForCategories(model, dfToSearch, df):
#     findOffenseCategory_udf = f.udf(findOffenseCategory, ArrayType(StringType()))
#     df.withColumn("offenseCategory", findOffenseCategory_udf(df.filteredFullText))
#     return df


# def topWords(words, vocabulary, num=5):
#     top = max(words[1])
#
#
# def mostUsedWords(df, vocabulary, num=5):
#     topWords_udf = f.udf(topWords, ArrayType(StringType()))
#     df = df.withColumn("fullTextCleaned", cleanFT_udf(df.fullText))

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



