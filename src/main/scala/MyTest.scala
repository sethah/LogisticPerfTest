import org.apache.spark.sql.{SparkSession, SQLContext}

object MyTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("spark session example")
      .getOrCreate()
    val sc = spark.sparkContext

    val whichAlgo = args.head
    val isML = whichAlgo match {
      case "ml" => true
      case "mllib" => false
      case _ => throw new IllegalArgumentException
    }
    val rows = args(1).toInt
    val cols = args(2).toInt

    val testResult = Tests.compareFeatureSizes(spark, isML, rows, cols)
    Tests.writeToCSV(testResult, "/Users/sethhendrickson/tests.txt")

    spark.stop()
  }

}
