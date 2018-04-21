from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer, NGram, CountVectorizer, MinHashLSH, \
    Word2Vec, StringIndexer, VectorIndexer
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, FloatType, BooleanType
import re




def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer")\
        .config("spark.driver.memory", "10G")\
        .getOrCreate()
    # sc.conf.set("spark.sql.shuffle.partitions", 6)
    # sc.conf.set("spark.executor.memory", "10G")
    # sc.conf.set("spark.driver.memory", "4G")
    # configMap: Map[String, String] = spark.conf.getAll()
    # print(sc.conf.get("spark.executor.memory"))
    # print(sc.conf.get("spark.executor.cores"))
    # print(sc.conf.get("spark.memory.fraction"))
    # print(sc.conf.get("spark.driver.memory"))

    sqlContext = SQLContext(sc)
    # print(sc.version)

    # main df
    cases = sc.read.json("../data/sentencingCases2.jsonl")
    df = cleanDf(cases)

    # read categorized csv
    categorizedCsv = sc.read.csv("../data/categorized.csv", header=True)
    # categorizedCsv = splitCategorizedCsvOffenceTypes(categorizedCsv).select('caseName', 'type')
    categorizedCsv =categorizedCsv.select('caseName', f.split(f.col("type"), " - ").alias('offenseType'), 'duration1', 'sentenceType1')

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
    rescaledData = idfModel.transform(result).filter(f.size('offenseKeywords') > 0)
    idfModelSearch = idf.fit(resultSearch)
    rescaledDataSearch = idfModelSearch.transform(resultSearch)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", seed=12345, numHashTables=20)
    modelMH = mh.fit(rescaledData)
    transformedData = modelMH.transform(rescaledData)

    modelMHSearch = mh.fit(rescaledDataSearch)
    transformedDataSearch = modelMH.transform(rescaledDataSearch)

    categorizedDf = modelMHSearch.approxSimilarityJoin(transformedDataSearch, transformedData, 0.89, distCol="JaccardDistance")
    distanceDf = categorizedDf.select([f.col('datasetA.term')] + [f.col('datasetB.caseID')] + [f.col("JaccardDistance")]) \
        .orderBy('caseID', 'JaccardDistance')

    # EVALUATE CATEGORIZATION AGAINST MANUAL CATEGORIZATION
    distanceDf = distanceDf.join(categorizedCsv, distanceDf.caseID == categorizedCsv.caseName).select('caseId', 'term', 'JaccardDistance', 'offenseType')
    distanceDf = distanceDf.groupBy('caseId').agg(f.collect_list('term').alias('predictedOffences'), f.collect_list('JaccardDistance').alias('JaccardDistances'), f.first('offenseType').alias('actualOffences'))
    distanceDf = distanceDf.filter(distanceDf.actualOffences[0] != "N/A").filter(distanceDf.actualOffences[0] != "multiple party sentence")
    # evaluate errors
    calcuateDifferenceInPredictedVsActualOffences_udf = f.udf(calcuateDifferenceInPredictedVsActualOffences, FloatType())
    distanceDf = distanceDf.withColumn("error", calcuateDifferenceInPredictedVsActualOffences_udf(distanceDf.predictedOffences, distanceDf.actualOffences))
    rmse = (distanceDf.groupBy().agg(f.sum('error')).collect()[0][0]/distanceDf.count())**(1.0/2)
    print("RMSE:", rmse)


    # QUANTIZATION OF SENTENCE DURATION
    # compute n-grams
    # ngram = NGram(n=3, inputCol="filteredFullText", outputCol="ngrams")
    # ngramDataFrame = ngram.transform(df)
    # ngramDataFrame = getTimeRelatedNGrams(ngramDataFrame)
    # categorizedNgrams = ngramDataFrame.join(categorizedCsv, ngramDataFrame.caseID == categorizedCsv.caseName).select('caseId', 'duration1', 'timeKeywords')
    # categorizedNgrams = categorizedNgrams.filter(categorizedNgrams.duration1 != "null").filter(f.size('timeKeywords') != 0)
    #
    # convertDurationsToDays_udf = f.udf(convertDurationsToDays, ArrayType(FloatType()))
    # mostCommon_udf = f.udf(mostCommon, FloatType())
    # quantified = categorizedNgrams\
    #     .withColumn("actualDays", convertDurationsToDays_udf(categorizedNgrams.duration1))\
    #     .withColumn("predictedDays", convertDurationsToDays_udf(categorizedNgrams.timeKeywords))
    # quantified = quantified.withColumn("actualDays", quantified.actualDays[0]).withColumn("predictedDays", mostCommon_udf(quantified.predictedDays))
    # quantified.select('caseId', 'actualDays', 'predictedDays').show(200)
    #
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="actualDays", predictionCol="predictedDays")
    # rmse = evaluator.evaluate(quantified)
    # print("Root-mean-square error = " + str(rmse))

    # categorizedNgrams = result.join(categorizedCsv, result.caseID == categorizedCsv.caseName).select('caseId', 'duration1', 'rawFeatures')
    #
    # # Index labels, adding metadata to the label column.
    # # Fit on whole dataset to include all labels in index.
    # labelIndexer = StringIndexer(inputCol="duration1", outputCol="indexedLabel", handleInvalid="skip").fit(categorizedNgrams)
    # # Automatically identify categorical features, and index them.
    # # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    # featureIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="indexedFeatures", maxCategories=100).fit(categorizedNgrams)
    #
    # # Split the data into training and test sets (30% held out for testing)
    # (trainingData, testData) = categorizedNgrams.randomSplit([0.7, 0.3])
    #
    # # Train a DecisionTree model.
    # dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    #
    # # Chain indexers and tree in a Pipeline
    # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    #
    # # Train model.  This also runs the indexers.
    # model = pipeline.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.select("prediction", "indexedLabel", 'caseID', 'duration1').show(100, truncate=False)


    # ngramDataFrame.select('caseID', 'timeKeywords').show(200, truncate=False)
    # SPLIT BY INDIGENOUS AND NON-INDIGENOUS
    categorizeByIndigeneity_udf = f.udf(categorizeByIndigeneity, BooleanType())
    ethnicityDf = df.withColumn("isIndigenous", categorizeByIndigeneity_udf(df.filteredFullText)).select('caseID', 'isIndigenous')
    ethnicityDf.show(200)

    # CLUSTER
    # combine all previous steps into single df


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
              ("sexual offences", ['sexual', 'sex', 'consent', 'nonconsensual', 'consensual', 'abuse', 'abused', 'rape', 'raped', 'incest', 'molester', 'touching', 'pants', 'penis', 'clothing', 'vagina', 'breasts', 'breast', 'grope', 'groped']),
              ("homicide", ['manslaughter', 'murder', 'death', 'weapon', 'kill', 'killing', 'deceased', 'meditated', 'premeditated', 'died', 'accidental']),
              ("terrorism", ['group', 'terrorism', 'terrorist']),
              ("drug offences", ['drug', 'drugs', 'trafficking', 'crack', 'cocaine', 'heroin', 'kilogram', 'kilograms', 'pound', 'pounds', 'ounces', 'grams', 'marijuana', 'intoxicating', 'methamphetamine', 'meth', 'lsd']),
              ("robbery", ['robbery', 'robbed', 'rob', 'stole', 'stolen', 'steal', 'stealing', 'break', 'enter', 'theft', '348', 'break', 'breaking', 'enter', 'entering']),
              ("weapon", ['weapon', 'firearm', 'firearms', 'pistol', 'rifle', 'knife', 'stabbed', 'stabbing', 'firing', 'fired', 'shooting', 'armed']),
              ("fraud", ['forgery', 'forging', 'fraud', 'impersonation', 'cheque', 'cheques', 'sum', 'financial', 'money', 'monies', 'deprivation', 'fraudulent', 'defraud', 'defrauded', 'defrauding', 'deceits', 'deceit', 'falsehood', 'breach', 'trust', 'con', 'artist', 'forgery']),
              ("child pornography", ['child', 'pornography', 'vile', 'disgusting', 'distribution', 'repulsive', 'underage']),
              ("mischief", ['mischief']),
              ("illegal possession", ['proceeds', 'possession']),
              ("escaping lawful custody", ['escape']),
              ("criminal organization", ['gang', 'mafia', 'hells', 'angels']),
              ("uttering threats", ['utter', 'uttered', 'uttering', 'threat', 'threats']),
              ("breach of trust", ['trust']),
              ("forcible confinement", ['confinement', 'kidnap', 'kidnapping']),
              ("regulatory offences", ['regulatory', 'municipal']),
              ("offences relating to public or peace officer", ['obstruction']),
              ("attempt offences", ['attempt', 'attempted', 'commit']),
              ("driving offences", ['253', 'driving', 'highway', 'traffic', 'suspended', 'hta', 'stunt', 'plates', 'careless', 'automobile', 'motor', 'vehicle', 'operate']),
              ("court-related offences", ['perjury', 'breaching', 'breach', 'breached', 'conditions', 'condition', 'order', 'comply', '731', '139', '145', '264', 'ltso']),
              ("tax offences", ['evading', 'evade', 'tax', 'income', 'taxation', 'hiding'])]

timeRelatedWords = ['day', 'days', 'month', 'months', 'year', 'years']

ageRelatedWords = ['old', 'age', 'aged', 'young', 'whe`n', 'victim', 'accused', 'person', 'ago', 'turned']

# sentenceRelatedWords = ['imprisonment', 'prison', 'sentenced', 'sentence', 'probation', 'incarceration', 'intermittent',
#                         'concurrent', 'reduced', 'incarceration', 'correctional', 'jail', 'supervised', 'custodial']
# only look at imprisonment
sentenceRelatedWords = ['imprisonment', 'prison', 'sentenced', 'sentence', 'incarceration', 'incarceration', 'correctional', 'jail', 'supervised', 'custodial']

indigenousRelatedWords = ['indigenous', 'aboriginal', 'gladue', 'ipeelee']


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


def timeKeywords(textArray):
    ngrams = []
    for ngram in textArray:
        words = ngram.split(' ')
        quantityFound = False
        for word in words:
            if word.isdigit():
                quantityFound = True
        if not quantityFound:
            continue
        sentenceRelatedWordFound = False
        for word in words:
            for keyword in sentenceRelatedWords:
                if word == keyword:
                    sentenceRelatedWordFound = True
                    break
        if not sentenceRelatedWordFound:
            continue
        ageRelatedWordFound = False
        for word in words:
            for keyword in ageRelatedWords:
                # skip any ngrams that are talking about someone's age
                if word == keyword:
                    ageRelatedWordFound = True
                    break
        if ageRelatedWordFound:
            continue
        for word in words:
            for keyword in timeRelatedWords:
                if word == keyword:
                    ngrams.append(ngram)
                    break
    return ngrams


def getTimeRelatedNGrams(df):
    timeKeywords_udf = f.udf(timeKeywords, ArrayType(StringType()))
    df = df.withColumn("timeKeywords", timeKeywords_udf(df.ngrams))
    return df


def convertDurationToDays(text):
    words = text.split(' ')
    quantity = 1.0
    for word in words:
        if word.isdigit():
            quantity = quantity * int(word)
        elif word == "month" or word == "months":
            quantity = quantity * 30
        elif word == "year" or word == "years":
            quantity = quantity * 365
    return quantity


def convertDurationsToDays(text):
    listOfDurations = []
    if isinstance(text, str):
        listOfDurations.append(convertDurationToDays(text))
    elif isinstance(text, list):
        for element in text:
            listOfDurations.append(convertDurationToDays(element))
    else:
        listOfDurations.append(0)
    return listOfDurations


def mostCommon(listOfDurations):
    return max(set(listOfDurations), key=listOfDurations.count)


def categorizeByIndigeneity(listWords):
    isIndigenous = False
    for word in listWords:
        if word in indigenousRelatedWords:
            isIndigenous = True
            break
    return isIndigenous


def calcuateDifferenceInPredictedVsActualOffences(predictedArray, actualArray):
    longer = []
    shorter = []
    if len(predictedArray) >= len(actualArray):
        longer = predictedArray
        shorter = actualArray
    else:
        longer = actualArray
        shorter = predictedArray
    found = 0
    for offenceL in longer:
        for offenceS in shorter:
            if offenceL == offenceS:
                found += 1
                break
    return ((found - len(longer))/len(longer))**2


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



