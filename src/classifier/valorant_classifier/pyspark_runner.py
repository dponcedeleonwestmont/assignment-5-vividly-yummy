from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Valorant").getOrCreate()
data = spark.read.csv("../../player_stats.csv", header=True, inferSchema=True)

processed_data = data.filter(data['fd'] > 5).select('number', 'player', 'map')

processed_data.show()

spark.stop()