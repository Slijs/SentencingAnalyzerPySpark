from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer, NGram, CountVectorizer, MinHashLSH, \
    Word2Vec, StringIndexer, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression, RandomForestRegressor, GBTRegressor, \
    IsotonicRegression, GeneralizedLinearRegression
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, FloatType, BooleanType
import re




def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer")\
        .config("spark.driver.memory", "10G")\
        .getOrCreate()

    # main df
    cases = sc.read.json("../data/sentencingCases2.jsonl")
    df = cleanDf(cases)

    # read categorized csv
    categorizedCsv = sc.read.csv("../data/categorized.csv", header=True)
    categorizedCsv =categorizedCsv.select('caseName', f.split(f.col("type"), " - ").alias('offenseType'), 'duration1', 'sentenceType1')

    # create the search df
    df = extractOffenseKeywords(df)
    df.cache()
    dfSearch = sc.createDataFrame(searchData, ["term", "offenseKeywords"])

    # CLASSIFICATION OF OFFENSE
    hashingTF = HashingTF(inputCol="offenseKeywords", outputCol="rawFeatures", numFeatures=1000)
    result = hashingTF.transform(df)
    resultSearch = hashingTF.transform(dfSearch)

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
    distanceDf = distanceDf.groupBy('caseID').agg(f.collect_list('term').alias('predictedOffences'), f.collect_list('JaccardDistance').alias('JaccardDistances'))
    distanceDf.cache()
    distanceDf.show()

    # EVALUATE CATEGORIZATION AGAINST MANUAL CATEGORIZATION
    distanceDfEval = distanceDf.join(categorizedCsv, distanceDf.caseID == categorizedCsv.caseName)
    distanceDfEval = distanceDfEval.filter(distanceDfEval.offenseType[0] != "N/A").filter(distanceDfEval.offenseType[0] != "multiple party sentence")
    calcuateDifferenceInPredictedVsActualOffences_udf = f.udf(calcuateDifferenceInPredictedVsActualOffences, FloatType())
    distanceDfEval = distanceDfEval.withColumn("error", calcuateDifferenceInPredictedVsActualOffences_udf(distanceDfEval.predictedOffences, distanceDfEval.offenseType))
    calcuateDifferenceInPredictedVsActualOffencesPercentage_udf = f.udf(calcuateDifferenceInPredictedVsActualOffencesPercentage, FloatType())
    distanceDfEval = distanceDfEval.withColumn("pctCorrect", calcuateDifferenceInPredictedVsActualOffencesPercentage_udf(distanceDfEval.predictedOffences, distanceDfEval.offenseType))
    distanceDfEval.select('caseID', 'predictedOffences', 'offenseType', 'JaccardDistances', 'error', 'pctCorrect').show(200, truncate=False)
    rmse = (distanceDfEval.groupBy().agg(f.sum('error')).collect()[0][0]/distanceDfEval.count())**(1.0/2)
    print("Offense category RMSE:", rmse)
    pctCorrectOffense = (distanceDfEval.groupBy().agg(f.sum('pctCorrect')).collect()[0][0] / distanceDfEval.count()) * 100
    print("Percentage of offenses correctly categorized: ", pctCorrectOffense)

    # QUANTIZATION OF SENTENCE DURATION
    # compute n-grams
    # ngram = NGram(n=3, inputCol="filteredFullText", outputCol="ngrams")
    # ngramDataFrame = ngram.transform(df)
    # ngramDataFrame = getTimeRelatedNGrams(ngramDataFrame)
    # convertDurationsToDays_udf = f.udf(convertDurationsToDays, ArrayType(FloatType()))
    # mostCommon_udf = f.udf(mostCommon, FloatType())
    # quantifiedDf = ngramDataFrame.withColumn("predictedDays", convertDurationsToDays_udf(ngramDataFrame.timeKeywords))
    # quantifiedDf = quantifiedDf.withColumn("predictedDays", mostCommon_udf(quantifiedDf.predictedDays))
    # quantifiedDf = quantifiedDf.select('caseID', 'timeKeywords', 'predictedDays')
    # quantifiedDf.cache()
    # # evaluate
    # quantifiedDfEval = quantifiedDf.join(categorizedCsv, ngramDataFrame.caseID == categorizedCsv.caseName).select('caseId', 'duration1', 'timeKeywords', 'predictedDays')
    # quantifiedDfEval = quantifiedDfEval.filter(quantifiedDfEval.duration1 != "null").filter(f.size('timeKeywords') != 0)
    # quantifiedDfEval = quantifiedDfEval.withColumn("actualDays", convertDurationsToDays_udf(quantifiedDfEval.duration1))
    # quantifiedDfEval = quantifiedDfEval.withColumn("actualDays", quantifiedDfEval.actualDays[0])
    # quantifiedDfEval = quantifiedDfEval.withColumn("correctlyPredicted", f.col('predictedDays') == f.col('actualDays'))
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="actualDays", predictionCol="predictedDays")
    # rmse = evaluator.evaluate(quantifiedDfEval)
    # print("Sentence duration RMSE:" + str(rmse))
    # numCorrect = quantifiedDfEval.groupBy().agg(f.sum(f.col("correctlyPredicted").cast("long"))).collect()[0][0]
    # totalCases = quantifiedDfEval.count()
    # print("numCorrect:", numCorrect)
    # print("totalCases:", totalCases)
    # pctCorrect = (numCorrect/totalCases)*100
    # print("Percentage of sentences correctly predicted: ", pctCorrect)

    # SPLIT BY INDIGENOUS AND NON-INDIGENOUS
    # categorizeByIndigeneity_udf = f.udf(categorizeByIndigeneity, BooleanType())
    # ethnicityDf = df.withColumn("isIndigenous", categorizeByIndigeneity_udf(df.filteredFullText)).select('caseID', 'isIndigenous')
    # ethnicityDf.cache()
    # ethnicityDf.show()

    # CLUSTER
    # combine all previous steps into single df
    # fullyCategorizedDf = distanceDf.join(quantifiedDf.join(ethnicityDf, 'caseID'), 'caseID')\
    #     .select('caseID', 'predictedOffences', 'predictedDays', 'isIndigenous')\
    #     .filter(f.col('predictedDays') != 0.0)
    # fullyCategorizedDf.cache()
    # # fullyCategorizedDf.show(200, truncate=False)
    #
    # fullyCategorizedDfHashingTF = HashingTF(inputCol="predictedOffences", outputCol="predictedOffencesVector", numFeatures=8)
    # fullyCategorizedDfFeaturized = fullyCategorizedDfHashingTF.transform(fullyCategorizedDf)
    # # fullyCategorizedDfFeaturized.show(200, truncate=False)
    #
    # # Split the data into training and test sets (30% held out for testing)
    # (trainingData, testData) = fullyCategorizedDfFeaturized.randomSplit([0.7, 0.3], seed=1234)
    #
    # # LINEAR REGRESSION
    # lr = LinearRegression(featuresCol="predictedOffencesVector", labelCol="predictedDays")
    #
    # # Fit the model
    # lrModel = lr.fit(trainingData)
    # predictions = lrModel.transform(testData)
    # predictions.show()
    #
    # # Print the coefficients and intercept for linear regression
    # # print("Coefficients: %s" % str(lrModel.coefficients))
    # # print("Intercept: %s" % str(lrModel.intercept))
    #
    # # Summarize the model over the training set and print out some metrics
    # trainingSummary = lrModel.summary
    # # print("numIterations: %d" % trainingSummary.totalIterations)
    # # print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    # # trainingSummary.residuals.show()
    # print("RMSE of linear regression: %f" % trainingSummary.rootMeanSquaredError)
    # # print("r2: %f" % trainingSummary.r2)
    #
    # # GENERALIZED LINEAR REGRESSION (GAMMA DISTRIBUTION)
    # glr = GeneralizedLinearRegression(featuresCol="predictedOffencesVector", labelCol="predictedDays", family="gaussian", link="identity", linkPredictionCol="p")
    # # Train model.  This also runs the indexer.
    # model = glr.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.show(200)
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of generalized linear regression (gaussian distribution) on test data = %g" % rmse)
    #
    # # DECISION TREE REGRESSION
    # # Train a DecisionTree model.
    # dt = DecisionTreeRegressor(featuresCol="predictedOffencesVector", labelCol="predictedDays")
    #
    # # Train model.  This also runs the indexer.
    # model = dt.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.show(200)
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of decision tree on test data = %g" % rmse)
    #
    # # RANDOM FOREST REGRESSION
    # rf = RandomForestRegressor(featuresCol="predictedOffencesVector", labelCol="predictedDays", numTrees=30)
    # # Train model.  This also runs the indexer.
    # model = rf.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.show(200)
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of random forest on test data = %g" % rmse)
    #
    # # GRADIENT BOOSTED TREE REGRESSION
    #
    # gbt = GBTRegressor(featuresCol="predictedOffencesVector", labelCol="predictedDays", maxIter=10)
    # # Train model.  This also runs the indexer.
    # model = gbt.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.show(200)
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of gradient boosted tree on test data = %g" % rmse)
    #
    # # ISOTONIC REGRESSION - widely varying sentences for groups of offenses make this a bad choice
    # iso = IsotonicRegression(featuresCol="predictedOffencesVector", labelCol="predictedDays")
    # model = iso.fit(trainingData)
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.show(200)
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of isotonic on test data = %g" % rmse)
    #
    # # K-MEANS
    # # Trains a k-means model.
    # kmeans = KMeans(featuresCol="predictedOffencesVector", k=10, seed=1234)
    # model = kmeans.fit(trainingData)
    #
    # # Make predictions
    # predictions = model.transform(testData)
    #
    # # Evaluate clustering by computing Silhouette score
    # evaluator = ClusteringEvaluator(featuresCol="predictedOffencesVector")
    #
    # silhouette = evaluator.evaluate(predictions)
    # print("Silhouette with squared euclidean distance = " + str(silhouette))
    #
    # # Shows the result.
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)
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
sentenceRelatedWords = ['imprisonment', 'prison', 'sentenced', 'sentence', 'incarceration', 'incarceration', 'correctional', 'custodial']

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
            if word.isdigit() and word != "0":
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
    if len(listOfDurations) == 0:
        return 0.0
    # sort to return the max sentence in case of ties (to better approximate concurrent sentences)
    # return max(sorted(set(listOfDurations), reverse=True), key=listOfDurations.count)
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


def calcuateDifferenceInPredictedVsActualOffencesPercentage(predictedArray, actualArray):
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
    return found/len(longer)

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



