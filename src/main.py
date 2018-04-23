from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer, NGram, CountVectorizer, MinHashLSH, \
    Word2Vec, StringIndexer, VectorIndexer, OneHotEncoderEstimator
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression, RandomForestRegressor, GBTRegressor, \
    IsotonicRegression, GeneralizedLinearRegression
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, FloatType, BooleanType, IntegerType
import re
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark_kmodes import *


def main():
    sc = SparkSession.builder.appName("SentencingAnalyzer")\
        .config("spark.driver.memory", "10G")\
        .getOrCreate()

    # main df
    cases = sc.read.json("../data/sentencingCases.jsonl")
    trainingCases = sc.read.json("../data/training.jsonl")
    df = cases.union(trainingCases)
    # CHANGE TO ONLY TRAINING CASES FOR EVALUATING METRICS OF OFFENSE TYPE CATEGORIZATION AND SENTENCE DURATION ESTIMATION
    df = cleanDf(df)
    # df.filter(f.col('caseID') == "2010oncj76").show(200, truncate=False)

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

    categorizedDf = modelMHSearch.approxSimilarityJoin(transformedDataSearch, transformedData, 0.98, distCol="JaccardDistance")
    distanceDf = categorizedDf.select([f.col('datasetA.term')] + [f.col('datasetB.caseID')] + [f.col("JaccardDistance")]) \
        .orderBy('caseID', 'JaccardDistance')
    distanceDf = distanceDf.groupBy('caseID').agg(f.collect_list('term').alias('predictedOffences'), f.collect_list('JaccardDistance').alias('JaccardDistances'))
    filterExtraOffences_udf = f.udf(filterExtraOffences, ArrayType(StringType()))
    distanceDf = distanceDf.withColumn("predictedOffences", filterExtraOffences_udf('predictedOffences', 'JaccardDistances')).select('caseID', 'predictedOffences')
    assignOffenceNumericalCategory_udf = f.udf(assignOffenceNumericalCategory, IntegerType())
    distanceDf = distanceDf.withColumn("predictedOffencesIndex", assignOffenceNumericalCategory_udf('predictedOffences'))
    distanceDf.cache()
    # distanceDf.show(200)

    # EVALUATE CATEGORIZATION AGAINST MANUAL CATEGORIZATION
    distanceDfEval = distanceDf.join(categorizedCsv, distanceDf.caseID == categorizedCsv.caseName)
    distanceDfEval = distanceDfEval.filter(distanceDfEval.offenseType[0] != "N/A").filter(distanceDfEval.offenseType[0] != "multiple party sentence")
    calcuateDifferenceInPredictedVsActualOffences_udf = f.udf(calcuateDifferenceInPredictedVsActualOffences, FloatType())
    distanceDfEval = distanceDfEval.withColumn("error", calcuateDifferenceInPredictedVsActualOffences_udf(distanceDfEval.predictedOffences, distanceDfEval.offenseType))
    calcuateDifferenceInPredictedVsActualOffencesPercentage_udf = f.udf(calcuateDifferenceInPredictedVsActualOffencesPercentage, FloatType())
    distanceDfEval = distanceDfEval.withColumn("pctCorrect", calcuateDifferenceInPredictedVsActualOffencesPercentage_udf(distanceDfEval.predictedOffences, distanceDfEval.offenseType))
    # distanceDfEval.select('caseID', 'predictedOffences', 'offenseType', 'error', 'pctCorrect').show(200, truncate=False)
    rmse = (distanceDfEval.groupBy().agg(f.sum('error')).collect()[0][0]/distanceDfEval.count())**(1.0/2)
    print("Offense category RMSE:", rmse)
    pctCorrectOffense = (distanceDfEval.groupBy().agg(f.sum('pctCorrect')).collect()[0][0] / distanceDfEval.count()) * 100
    print("Percentage of offenses correctly categorized: ", pctCorrectOffense)
    print("Num cases categorized:", distanceDf.count())
    print("Total cases available:", df.count())

    # QUANTIZATION OF SENTENCE DURATION
    # compute n-grams
    ngram = NGram(n=3, inputCol="filteredFullText", outputCol="ngrams")
    ngramDataFrame = ngram.transform(df)
    ngramDataFrame = getTimeRelatedNGrams(ngramDataFrame)
    convertDurationsToDays_udf = f.udf(convertDurationsToDays, ArrayType(FloatType()))
    mostCommon_udf = f.udf(mostCommon, FloatType())
    quantifiedDf = ngramDataFrame.withColumn("predictedDays", convertDurationsToDays_udf(ngramDataFrame.timeKeywords))
    quantifiedDf = quantifiedDf.withColumn("predictedDays", mostCommon_udf(quantifiedDf.predictedDays))
    quantifiedDf = quantifiedDf.select('caseID', 'timeKeywords', 'predictedDays')
    quantifiedDf.cache()
    # evaluate
    quantifiedDfEval = quantifiedDf.join(categorizedCsv, ngramDataFrame.caseID == categorizedCsv.caseName).select('caseId', 'duration1', 'timeKeywords', 'predictedDays')
    quantifiedDfEval = quantifiedDfEval.filter(quantifiedDfEval.duration1 != "null").filter(f.size('timeKeywords') != 0)
    quantifiedDfEval = quantifiedDfEval.withColumn("actualDays", convertDurationsToDays_udf(quantifiedDfEval.duration1))
    quantifiedDfEval = quantifiedDfEval.withColumn("actualDays", quantifiedDfEval.actualDays[0])
    quantifiedDfEval = quantifiedDfEval.withColumn("correctlyPredicted", f.col('predictedDays') == f.col('actualDays'))
    # quantifiedDfEval.select('caseId', 'predictedDays', 'actualDays', 'timeKeywords').show(200, truncate=False)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="actualDays", predictionCol="predictedDays")
    rmse = evaluator.evaluate(quantifiedDfEval)
    print("Sentence duration RMSE:" + str(rmse))
    numCorrect = quantifiedDfEval.groupBy().agg(f.sum(f.col("correctlyPredicted").cast("long"))).collect()[0][0]
    totalCases = quantifiedDfEval.count()
    print("numCorrect:", numCorrect)
    print("totalCases:", totalCases)
    pctCorrect = (numCorrect/totalCases)*100
    print("Percentage of sentences correctly predicted: ", pctCorrect)

    # SPLIT BY INDIGENOUS AND NON-INDIGENOUS
    categorizeByIndigeneity_udf = f.udf(categorizeByIndigeneity, IntegerType())
    ethnicityDf = df.withColumn("isIndigenous", categorizeByIndigeneity_udf(df.filteredFullText)).select('caseID', 'isIndigenous')
    ethnicityDf.cache()
    # ethnicityDf.show()

    # COMBINE
    # combine all previous steps into single df
    fullyCategorizedDf = distanceDf.join(quantifiedDf.join(ethnicityDf, 'caseID'), 'caseID')\
        .select('caseID', 'predictedOffences', 'predictedOffencesIndex', 'predictedDays', 'isIndigenous')\
        .filter(f.col('predictedDays') != 0.0)
    fullyCategorizedDf = fullyCategorizedDf.na.drop()
    fullyCategorizedDf.cache()
    print("Total cases fully classified: ", fullyCategorizedDf.count())
    # fullyCategorizedDf.show(200, truncate=False)

    # fullyCategorizedDfHashingTF = HashingTF(inputCol="predictedOffences", outputCol="predictedOffencesVector", numFeatures=22)
    # fullyCategorizedDfFeaturized = fullyCategorizedDfHashingTF.transform(fullyCategorizedDf)
    # fullyCategorizedDfFeaturized.show(200, truncate=False)

    # fit a CountVectorizerModel from the corpus.
    # cv = CountVectorizer(inputCol="predictedOffences", outputCol="predictedOffencesVector", vocabSize=22, minDF=0.0)
    #
    # model = cv.fit(fullyCategorizedDf)
    #
    # result = model.transform(fullyCategorizedDf)
    # result.show(truncate=False)

    # custom vectorizer
    vectorizePredictedOffences_udf = f.udf(vectorizePredictedOffences, ArrayType(IntegerType()))
    list_to_vector_udf = f.udf(lambda l: Vectors.dense(l), VectorUDT())
    fullyCategorizedDfFeaturized = fullyCategorizedDf\
        .withColumn('predictedOffencesVector', vectorizePredictedOffences_udf('predictedOffences'))\
        .withColumn('predictedOffencesVector', list_to_vector_udf('predictedOffencesVector'))
    # fullyCategorizedDfFeaturized.show(truncate=False)

    # encoder = OneHotEncoderEstimator(inputCols=["predictedOffencesIndex", "isIndigenous"], outputCols=["predictedOffencesVector", "isIndigenousVector"])
    # model = encoder.fit(fullyCategorizedDf)
    # fullyCategorizedDfFeaturized = model.transform(fullyCategorizedDf)
    # fullyCategorizedDfFeaturized.show(truncate=False)

    # Split the data into indigenous and non-indigenous sets (30% held out for testing)
    indigenousOffendersDf = fullyCategorizedDfFeaturized.filter(fullyCategorizedDfFeaturized.isIndigenous == 1)
    print("Number of cases involving indigenous offenders:", indigenousOffendersDf.count())
    nonIndigenousOffendersDf = fullyCategorizedDfFeaturized.filter(fullyCategorizedDfFeaturized.isIndigenous == 0)
    print("Number of cases involving non-indigenous offenders:", nonIndigenousOffendersDf.count())

    # meanByOffenceTypeIndigenousOffendersRows = indigenousOffendersDf\
    #     .groupBy('predictedOffencesIndex')\
    #     .agg({'predictedDays': 'mean'}).collect()
    # print("\nMean of offences for indigenous offenders:")
    # meanByOffenceTypeIndigenousOffenders = []
    # for index, meanOfOffence in enumerate(meanByOffenceTypeIndigenousOffendersRows):
    #     type = searchData[meanOfOffence[0]][0]
    #     mean = meanOfOffence[1]
    #     meanByOffenceTypeIndigenousOffenders.append((type, mean))
    #     print(searchData[meanOfOffence[0]][0], meanOfOffence[1])
    #
    # meanByOffenceTypeNonIndigenousOffendersRows = nonIndigenousOffendersDf\
    #     .groupBy('predictedOffencesIndex')\
    #     .agg({'predictedDays': 'mean'}).collect()
    # meanByOffenceTypeNonIndigenousOffenders = []
    # for index, meanOfOffence in enumerate(meanByOffenceTypeNonIndigenousOffendersRows):
    #     type = searchData[meanOfOffence[0]][0]
    #     mean = meanOfOffence[1]
    #     meanByOffenceTypeNonIndigenousOffenders.append((type, mean))
    #     print(searchData[meanOfOffence[0]][0], meanOfOffence[1])
    #
    # listOffences = fullyCategorizedDfFeaturized.select(fullyCategorizedDfFeaturized.predictedOffences[0]).collect()
    # x = []
    # for offence in listOffences:
    #     x.append(offence[0])
    # y = fullyCategorizedDfFeaturized.select('predictedDays').collect()
    # plt.figure(figsize=(8, 10))
    # plt.scatter(x, y, label="Case")
    # plt.scatter([x[0] for x in meanByOffenceTypeNonIndigenousOffenders], [x[1] for x in meanByOffenceTypeNonIndigenousOffenders], label="Mean (non-Indigenous)", s=90, c="cyan")
    # plt.scatter([x[0] for x in meanByOffenceTypeIndigenousOffenders], [x[1] for x in meanByOffenceTypeIndigenousOffenders], label="Mean (Indigenous)", s=90, c="magenta")
    # plt.xticks(rotation=-90)
    # plt.ylabel("Days incarcerated")
    # plt.xlabel("Offense type")
    # plt.legend()
    # plt.title("Estimated cases and sentence durations from full text analysis")
    # plt.show()

    # VISUALIZE CATEGORIZED DATA


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
    glr = GeneralizedLinearRegression(featuresCol="predictedOffencesVector", labelCol="predictedDays", family="gaussian", link="identity", linkPredictionCol="p")
    # # Train model.  This also runs the indexer.
    modelNonIndigenous = glr.fit(nonIndigenousOffendersDf)
    # Make predictions.
    predictionsNonIndigenous = modelNonIndigenous.transform(nonIndigenousOffendersDf)
    # Select example rows to display.
    # predictions.show()
    evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictionsNonIndigenous)
    print("RMSE of generalized linear regression (gaussian distribution) on non-indigenous data = %g" % rmse)

    meanByOffenceTypeNonIndigenousOffendersRows = predictionsNonIndigenous\
        .groupBy('predictedOffencesIndex')\
        .agg({'prediction': 'mean'}).collect()
    meanByOffenceTypeNonIndigenousOffenders = []
    for index, meanOfOffence in enumerate(meanByOffenceTypeNonIndigenousOffendersRows):
        type = searchData[meanOfOffence[0]][0]
        mean = meanOfOffence[1]
        meanByOffenceTypeNonIndigenousOffenders.append((type, mean))
        print(searchData[meanOfOffence[0]][0], meanOfOffence[1])

    # Train model.  This also runs the indexer.
    modelIndigenous = glr.fit(indigenousOffendersDf)
    # Make predictions.
    predictionsIndigenous = modelIndigenous.transform(indigenousOffendersDf)
    # Select example rows to display.
    # predictions.show()
    rmse = evaluator.evaluate(predictionsIndigenous)
    print("RMSE of generalized linear regression (gaussian distribution) on indigenous data = %g" % rmse)

    meanByOffenceTypeIndigenousOffendersRows = predictionsIndigenous\
        .groupBy('predictedOffencesIndex')\
        .agg({'prediction': 'mean'}).collect()
    print("\nMean of offences for indigenous offenders:")
    meanByOffenceTypeIndigenousOffenders = []
    for index, meanOfOffence in enumerate(meanByOffenceTypeIndigenousOffendersRows):
        type = searchData[meanOfOffence[0]][0]
        mean = meanOfOffence[1]
        meanByOffenceTypeIndigenousOffenders.append((type, mean))
        print(searchData[meanOfOffence[0]][0], meanOfOffence[1])
    #
    # plt.figure(figsize=(8, 10))
    # plt.scatter([x[0] for x in meanByOffenceTypeNonIndigenousOffenders], [x[1] for x in meanByOffenceTypeNonIndigenousOffenders], label="Mean (non-Indigenous)", s=90, c="cyan")
    # plt.scatter([x[0] for x in meanByOffenceTypeIndigenousOffenders], [x[1] for x in meanByOffenceTypeIndigenousOffenders], label="Mean (Indigenous)", s=90, c="magenta")
    # plt.xticks(rotation=-90)
    # plt.ylabel("Days incarcerated")
    # plt.xlabel("Offense type")
    # plt.title("Generalized linear regression (gaussian distribution) predictions")
    # plt.legend()
    # plt.show()


    meanByOffenceTypeNonIndigenousOffendersDict = dict(meanByOffenceTypeNonIndigenousOffenders)
    meanByOffenceTypeIndigenousOffendersDict = dict(meanByOffenceTypeIndigenousOffenders)
    for offenceType in searchData:
        if offenceType[0] not in meanByOffenceTypeIndigenousOffendersDict:
            meanByOffenceTypeIndigenousOffenders.append((offenceType[0], 0))
        if offenceType[0] not in meanByOffenceTypeNonIndigenousOffendersDict:
            meanByOffenceTypeNonIndigenousOffenders.append((offenceType[0], 0))

    numCategories = len(meanByOffenceTypeIndigenousOffenders)
    ind = np.arange(numCategories)
    plotBarWidth = 0.3
    plt.figure(figsize=(8, 10))
    plt.bar(ind+plotBarWidth, [x[1] for x in meanByOffenceTypeNonIndigenousOffenders], plotBarWidth, label="Mean (non-Indigenous)", color="cyan")
    plt.bar(ind, [x[1] for x in meanByOffenceTypeIndigenousOffenders], plotBarWidth, label="Mean (Indigenous)", color="magenta")
    plt.xticks(ind + plotBarWidth / 2, [x[0] for x in meanByOffenceTypeIndigenousOffenders], rotation=-90)
    plt.ylabel("Days incarcerated")
    plt.xlabel("Offense type")
    plt.title("Generalized linear regression (gaussian distribution) predictions")
    plt.legend()
    plt.show()


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
    # predictions.show()
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of decision tree on test data = %g" % rmse)
    #
    # # RANDOM FOREST REGRESSION
    # rf = RandomForestRegressor(featuresCol="predictedOffencesVector", labelCol="predictedDays", numTrees=30)
    # modelNonIndigenous = rf.fit(nonIndigenousOffendersDf)
    # # Make predictions.
    # predictionsNonIndigenous = modelNonIndigenous.transform(nonIndigenousOffendersDf)
    # # Select example rows to display.
    # # predictions.show()
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictionsNonIndigenous)
    # print("RMSE of generalized linear regression (gaussian distribution) on non-indigenous data = %g" % rmse)
    #
    # meanByOffenceTypeNonIndigenousOffendersRows = predictionsNonIndigenous\
    #     .groupBy('predictedOffencesIndex')\
    #     .agg({'prediction': 'mean'}).collect()
    # meanByOffenceTypeNonIndigenousOffenders = []
    # for index, meanOfOffence in enumerate(meanByOffenceTypeNonIndigenousOffendersRows):
    #     type = searchData[meanOfOffence[0]][0]
    #     mean = meanOfOffence[1]
    #     meanByOffenceTypeNonIndigenousOffenders.append((type, mean))
    #     print(searchData[meanOfOffence[0]][0], meanOfOffence[1])
    #
    # # Train model.  This also runs the indexer.
    # modelIndigenous = rf.fit(indigenousOffendersDf)
    # # Make predictions.
    # predictionsIndigenous = modelIndigenous.transform(indigenousOffendersDf)
    # # Select example rows to display.
    # predictionsIndigenous.show()
    # rmse = evaluator.evaluate(predictionsIndigenous)
    # print("RMSE of generalized linear regression (gaussian distribution) on indigenous data = %g" % rmse)
    #
    # meanByOffenceTypeIndigenousOffendersRows = predictionsIndigenous\
    #     .groupBy('predictedOffencesIndex')\
    #     .agg({'prediction': 'mean'}).collect()
    # print("\nMean of offences for indigenous offenders:")
    # meanByOffenceTypeIndigenousOffenders = []
    # for index, meanOfOffence in enumerate(meanByOffenceTypeIndigenousOffendersRows):
    #     type = searchData[meanOfOffence[0]][0]
    #     mean = meanOfOffence[1]
    #     meanByOffenceTypeIndigenousOffenders.append((type, mean))
    #     print(searchData[meanOfOffence[0]][0], meanOfOffence[1])
    # meanByOffenceTypeNonIndigenousOffendersDict = dict(meanByOffenceTypeNonIndigenousOffenders)
    # meanByOffenceTypeIndigenousOffendersDict = dict(meanByOffenceTypeIndigenousOffenders)
    # for offenceType in searchData:
    #     if offenceType[0] not in meanByOffenceTypeIndigenousOffendersDict:
    #         meanByOffenceTypeIndigenousOffenders.append((offenceType[0], 0))
    #     if offenceType[0] not in meanByOffenceTypeNonIndigenousOffendersDict:
    #         meanByOffenceTypeNonIndigenousOffenders.append((offenceType[0], 0))
    #
    # numCategories = len(meanByOffenceTypeNonIndigenousOffenders)
    # ind = np.arange(numCategories)
    # plotBarWidth = 0.3
    # plt.figure(figsize=(8, 10))
    # plt.bar(ind, [x[1] for x in meanByOffenceTypeIndigenousOffenders], plotBarWidth, label="Mean (Indigenous)", color="magenta")
    # plt.bar(ind+plotBarWidth, [x[1] for x in meanByOffenceTypeNonIndigenousOffenders], plotBarWidth, label="Mean (non-Indigenous)", color="cyan")
    # plt.xticks(ind + plotBarWidth / 2, [x[0] for x in meanByOffenceTypeNonIndigenousOffenders], rotation=-90)
    # plt.ylabel("Days incarcerated")
    # plt.xlabel("Offense type")
    # plt.title("Random forest regression predictions")
    # plt.legend()
    # plt.show()

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
    # predictions.show()
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
    # predictions.show()
    #
    # evaluator = RegressionEvaluator(labelCol="predictedDays", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) of isotonic on test data = %g" % rmse)

    # K-MEANS
    # Trains a k-means model.
    # kmeans = KMeans(featuresCol="predictedOffencesVector", k=10, seed=1234)
    # model = kmeans.fit(fullyCategorizedDfFeaturized)
    #
    # # Make predictions
    # predictions = model.transform(fullyCategorizedDfFeaturized)
    # predictions.show(500, truncate=False)
    #
    # # Evaluate clustering by computing Silhouette score
    # evaluator = ClusteringEvaluator(featuresCol="predictedOffencesVector")
    #
    # silhouette = evaluator.evaluate(predictions)
    # print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)

    # K-MODES

    # n_clusters = 20
    # max_iter = 10
    # method = EnsembleKModes(n_clusters, max_iter)
    # model = method.fit(fullyCategorizedDfFeaturized.select('predictedOffencesIndex', 'predictedDays').rdd)

    # print(model.clusters)
    # print(method.mean_cost)
    # predictions = method.predictions
    # datapoints = method.indexed_rdd
    # combined = datapoints.zip(predictions)
    # print(combined.take(10))

    # centroids = model.clusters

    # model.predict(rdd).take(5)
    # model.predict(sc.parallelize(['e', 'e', 'f', 'e', 'e', 'f', 'g', 'e', 'f', 'e'])).collect()

    # VISUALIZE RESULTS
    # x = predictions.select('isIndigenous').collect()
    # listOffences = predictions.select(predictions.predictedOffences[0]).collect()
    # x = []
    # for offence in listOffences:
    #     x.append(offence[0])
    # y = predictions.select('predictedDays').collect()
    # plt.figure(figsize=(8, 10))
    # plt.scatter(x, y)
    # plt.scatter([x[0] for x in centroids], [x[1] for x in centroids], s=90, c="red")
    # plt.xticks(rotation=-80)
    # plt.show()


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
              ("homicide", ['manslaughter', 'murder', 'death', 'dead', 'weapon', 'kill', 'killing', 'deceased', 'meditated', 'premeditated', 'died', 'accidental', 'pronounced']),
              ("terrorism", ['group', 'terrorism', 'terrorist']),
              ("drug offences", ['drug', 'drugs', 'trafficking', 'crack', 'cocaine', 'heroin', 'kilogram', 'kilograms', 'pound', 'pounds', 'ounces', 'grams', 'marijuana', 'intoxicating', 'methamphetamine', 'meth', 'lsd']),
              ("robbery", ['robbery', 'robbed', 'rob', 'stole', 'stolen', 'steal', 'stealing', 'break', 'enter', 'theft', '348', 'break', 'breaking', 'enter', 'entering']),
              ("weapon", ['weapon', 'firearm', 'firearms', 'pistol', 'rifle', 'knife', 'knives', 'puncture', 'stab', 'stabbed', 'stabbing', 'firing', 'fired', 'shooting', 'armed']),
              ("fraud", ['forgery', 'forging', 'fraud', 'impersonation', 'cheque', 'cheques', 'sum', 'financial', 'money', 'monies', 'deprivation', 'fraudulent', 'defraud', 'defrauded', 'defrauding', 'deceits', 'deceit', 'falsehood', 'breach', 'trust', 'con', 'artist', 'forgery']),
              ("child pornography", ['child', 'pornography', 'vile', 'disgusting', 'distribution', 'repulsive', 'underage']),
              ("mischief", ['mischief', 'mislead', 'misled']),
              ("illegal possession", ['proceeds', 'possession']),
              ("escaping lawful custody", ['escape']),
              ("criminal organization", ['gang', 'mafia', 'hells', 'angels']),
              ("uttering threats", ['utter', 'uttered', 'uttering', 'threat', 'threats']),
              ("breach of trust", ['trust']),
              ("forcible confinement", ['forcible', 'confinement', 'kidnap', 'kidnapping']),
              ("regulatory offences", ['regulatory', 'municipal']),
              ("offences against police", ['obstruction', '129', 'peace', 'officer']),
              ("attempt offences", ['attempt', 'attempted', 'commit']),
              ("driving offences", ['253', 'driving', 'highway', 'traffic', 'suspended', 'hta', 'stunt', 'plates', 'careless', 'automobile', 'motor', 'vehicle', 'operate']),
              ("court-related offences", ['perjury', 'breaching', 'breach', 'breached', 'conditions', 'condition', 'order', 'comply', 'curfew', 'terms' '731', '139', '145', '264', 'ltso']),
              ("tax offences", ['evading', 'evade', 'tax', 'income', 'taxation', 'hiding'])]

timeRelatedWords = ['day', 'days', 'month', 'months', 'year', 'years']

ageRelatedWords = ['old', 'age', 'aged', 'young', 'whe`n', 'victim', 'accused', 'person', 'ago', 'turned']

# sentenceRelatedWords = ['imprisonment', 'prison', 'sentenced', 'sentence', 'probation', 'incarceration', 'intermittent',
#                         'concurrent', 'reduced', 'incarceration', 'correctional', 'jail', 'supervised', 'custodial']
# only look at imprisonment
sentenceRelatedWords = ['imprisonment', 'prison', 'sentenced', 'sentence', 'incarceration', 'incarceration', 'correctional', 'custodial', 'custody', 'impose']

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


def filterExtraOffences(listOffences, listDistances):
    if len(listOffences) < 2:
        return listOffences
    else:
        newListOffences = []
        newListOffences.append(listOffences[0])
        firstDistance = listDistances[0]
        for index, distance in enumerate(listDistances[1:], 1):
            if len(newListOffences) >= 1:
                break
            else:
                if distance - firstDistance < 0.04:
                    newListOffences.append(listOffences[index])
                else:
                    break
        return newListOffences



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
    if isIndigenous:
        return 1
    else:
        return 0


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
    # return found/len(longer)
    if found > 0:
        return 1.0
    else:
        return 0.0

def vectorizePredictedOffences(listOffences):
    vector = [0] * len(searchData)
    for offence in listOffences:
        for index, category in enumerate(searchData):
            if offence == category[0]:
                vector[index] = 1
                break
    return vector


def assignOffenceNumericalCategory(listOffences):
    for index, category in enumerate(searchData):
        if listOffences[0] == category[0]:
            return index


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



