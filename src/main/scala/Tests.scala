import org.apache.spark.mllib.classification.{LogisticRegressionModel => OldLogisticRegressionModel}
import org.apache.spark.sql.SparkSession

import java.io.StringWriter
import scala.collection.JavaConversions._
import java.io.FileWriter
import java.io.BufferedWriter


case class TestResult(settings: scala.collection.mutable.HashMap[String, String]) {

  def csvRows: String = {
    val argString = settings.foldLeft("") { case (s, (k, v)) =>
      s + k + "->" + v + "|"
    }
    argString
  }
}

object Tests {
  def compareSizes(spark: SparkSession, isML: Boolean): TestResult = {
    val seed = 4L
    val mlTest = new LogisticTest()
    val mllibTest = new OldMultinomialLogisticTest()
    val sizes = Array(500000000)
    val mlTimings = Array.ofDim[Double](sizes.length)
    val mllibTimings = Array.ofDim[Double](sizes.length)
    val mlMetrics = Array.ofDim[Double](sizes.length)
    val mllibMetrics = Array.ofDim[Double](sizes.length)
    val numClasses = 3
    val numFeatures = 20
    val regParam = 0.0
    val elasticNetParam = 0.0
    val standardization = true
    val fitIntercept = true
    val args = new scala.collection.mutable.HashMap[String, String]
    args += ("regParam" -> regParam.toString, "elasticNetParam" -> elasticNetParam.toString,
      "fitIntercept" -> fitIntercept.toString, "standardization" -> standardization.toString,
      "numClasses" -> numClasses.toString, "numFeatures" -> numFeatures.toString)
    val args1 = new scala.collection.mutable.HashMap[String, String]
    args1 += ("regParam" -> regParam.toString, "elasticNetParam" -> elasticNetParam.toString,
      "fitIntercept" -> fitIntercept.toString, "standardization" -> standardization.toString,
      "numClasses" -> (numClasses).toString, "numFeatures" -> numFeatures.toString)
    sizes.indices.foreach { i =>
      val rdd = MultinomialDataGenerator.makeData(spark, numClasses, numFeatures,
        fitIntercept, sizes(i), seed)
      val ml = true
      if (!ml) {
        val mllibTraining = mllibTest.convertData(spark, rdd)
        val (mllibModel, mllibTime) = mllibTest.runTest(mllibTraining, args1, 1)
        mllibTimings(i) = mllibTime
        mllibMetrics(i) = mllibTest.validate(mllibModel, mllibTest.convertData(spark, rdd))
      } else {
        val mlTraining = mlTest.convertData(spark, rdd)//.cache()
        val (mlModel, mlTime) = mlTest.runTest(mlTraining, args, 1)
        mlTimings(i) = mlTime
        mlMetrics(i) = mlTest.validate(mlModel, mlTest.convertData(spark, rdd))
//        mlTraining.unpersist()
      }
      println("---")
    }
//    println("ML:", mlTimings.mkString(","))
//    println("MLLIB:", mllibTimings.mkString(","))
    TestResult(args)
  }

  def compareSizesWithL2(spark: SparkSession): TestResult = {
    val seed = 4L
    val mlTest = new LogisticTest()
    val mllibTest = new OldMultinomialLogisticTest()
    val sizes = Array(1000, 10000, 100000, 1000000, 10000000)
    //    val sizes = Array(1000, 10000)
    val mlTimings = Array.ofDim[Double](sizes.length)
    val mllibTimings = Array.ofDim[Double](sizes.length)
    val mlMetrics = Array.ofDim[Double](sizes.length)
    val mllibMetrics = Array.ofDim[Double](sizes.length)
    val numClasses = 3
    val numFeatures = 20
    val regParam = 0.1
    val elasticNetParam = 0.0
    val standardization = true
    val fitIntercept = true
    val args = new scala.collection.mutable.HashMap[String, String]
    args += ("regParam" -> regParam.toString, "elasticNetParam" -> elasticNetParam.toString,
      "fitIntercept" -> fitIntercept.toString, "standardization" -> standardization.toString,
      "numClasses" -> numClasses.toString, "numFeatures" -> numFeatures.toString)
    sizes.indices.foreach { i =>
      val rdd = MultinomialDataGenerator.makeData(spark, numClasses, numFeatures,
        fitIntercept, sizes(i), seed)
      val Array(training, test) = rdd.randomSplit(Array(0.8, 0.2))

      val mllibTraining = mllibTest.convertData(spark, training).cache()
      val (mllibModel, mllibTime) = mllibTest.runTest(mllibTraining, args, 3)
      mllibTimings(i) = mllibTime
      mllibMetrics(i) = mllibTest.validate(mllibModel, mllibTest.convertData(spark, test))
      mllibTraining.unpersist()
      val mlTraining = mlTest.convertData(spark, training).cache()
      val (mlModel, mlTime) = mlTest.runTest(mlTraining, args, 3)
      mlTimings(i) = mlTime
      mlMetrics(i) = mlTest.validate(mlModel, mlTest.convertData(spark, test))
      mlTraining.unpersist()
    }
    println("ML:", mlTimings.mkString(","))
    println("MLLIB:", mllibTimings.mkString(","))
    TestResult(args)
  }

  def compareFeatureSizes(spark: SparkSession,
                          isML: Boolean, rows: Int, cols: Int): TestResult = {
    val seed = 4L
//    val mlTest = new MultinomialLogisticTest()
    val mlTest = new LogisticTest()
    val mllibTest = new OldMultinomialLogisticTest()
    val size = rows
    val numFeatures = Array(cols)
    val mlTimings = Array.ofDim[Double](numFeatures.length)
    val mllibTimings = Array.ofDim[Double](numFeatures.length)
    val mlMetrics = Array.ofDim[Double](numFeatures.length)
    val mllibMetrics = Array.ofDim[Double](numFeatures.length)
//    val numClasses = if (isML) 5 else 5
    val numClasses = 2
    val regParam = 0.0
    val elasticNetParam = 0.0
    val maxIter = 30
    val standardization = true
    val fitIntercept = true
    val args = new scala.collection.mutable.HashMap[String, String]
    args += ("regParam" -> regParam.toString, "elasticNetParam" -> elasticNetParam.toString,
      "fitIntercept" -> fitIntercept.toString, "standardization" -> standardization.toString,
      "numClasses" -> numClasses.toString, "numPoints" -> size.toString,
      "maxIter" -> maxIter.toString, "numFeatures" -> numFeatures.head.toString)
    numFeatures.indices.foreach { i =>
      val rdd = MultinomialDataGenerator.makeData(spark, numClasses, numFeatures(i),
        fitIntercept, size, seed)
      val Array(training, test) = rdd.randomSplit(Array(0.8, 0.2))
      val counts = training.map(lp => lp.label).countByValue()
      if (isML) {
        val mlTraining = mlTest.convertData(spark, training)//.cache()
        val (mlModel, mlTime) = mlTest.runTest(mlTraining, args, 1)
        args += ("time" -> mlTime.toString)
        val mlMetric = mlTest.validate(mlModel, mlTest.convertData(spark, test))
        args += ("metric" -> mlMetric.toString)
        args += ("algo" -> "ml")
//        mlTraining.unpersist()
      } else {
        val mllibTraining = mllibTest.convertData(spark, training)//.cache()
        val (mllibModel, mllibTime) = mllibTest.runTest(mllibTraining, args, 1)
        args += ("time" -> mllibTime.toString)
        val mllibMetric = mllibTest.validate(mllibModel, mllibTest.convertData(spark, test))
        args += ("metric" -> mllibMetric.toString)
        args += ("algo" -> "mllib")
//        mllibTraining.unpersist()
      }
      println("---")
    }
    TestResult(args)
  }

  def writeToCSV(tests: TestResult, filePath: String = "tests.csv"): Unit = {
    scala.tools.nsc.io.File(filePath).appendAll(tests.csvRows + "\n")
  }
}
