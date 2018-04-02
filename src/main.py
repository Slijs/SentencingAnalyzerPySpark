from pyspark.sql import SparkSession

# Constants
# myMaxIter = 5
# myRank = 70
# myRegParam = 0.01
# myColdStartStrategy = 'drop'

spark = SparkSession.builder.appName("SentencingAnalyzer").getOrCreate()

print(spark.version)

cases = spark.read.json("../data/sentencingCases.jsonl")

# cases = spark.read.option("multiLine", True).option("mode", "PERMISSIVE").json("../data/sentencingCases.json")

cases.show()