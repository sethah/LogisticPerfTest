import org.apache.spark.ml.classification.{MultinomialLogisticRegression, MultinomialLogisticRegressionModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, Instance}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel => OldLogisticRegressionModel}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors, Vector => OldVector}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame}

abstract class PerfTest[M, DataType] {

  def convertData(spark: SparkSession, rdd: RDD[LabeledPoint]): DataType

  def runTest(
      data: DataType,
      args: scala.collection.mutable.HashMap[String, String],
      bestOf: Int): (M, Double)

  def validate(model: M, testData: DataType): Double

}

class MultinomialLogisticTest extends PerfTest[MultinomialLogisticRegressionModel, DataFrame] {

  def convertData(spark: SparkSession, rdd: RDD[LabeledPoint]): DataFrame = {
    spark.createDataFrame(rdd)
  }

  def runTest(
    data: DataFrame,
    args: scala.collection.mutable.HashMap[String, String],
    bestOf: Int): (MultinomialLogisticRegressionModel, Double) = {
    val regParam = args.getOrElse("regParam", "0.0").toDouble
    val elasticNetParam = args.getOrElse("elasticNetParam", "0.0").toDouble
    val standardization = args.getOrElse("standardization", "true").toBoolean
    val fitIntercept = args.getOrElse("fitIntercept", "true").toBoolean
    val maxIter = args.getOrElse("maxIter", "100").toInt
    val mlr = new MultinomialLogisticRegression()
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setFitIntercept(fitIntercept)
      .setStandardization(standardization)
      .setMaxIter(maxIter)

    val results = (0 until bestOf).map { i =>
      val t0 = System.currentTimeMillis()
      val model = mlr.fit(data)
      val t1 = System.currentTimeMillis()
      (model, (t1 - t0) / 1e3)
    }
    results.minBy(_._2)
  }

  def validate(model: MultinomialLogisticRegressionModel, testData: DataFrame): Double = {
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val transformed = model.transform(testData)
    evaluator.evaluate(transformed)
  }
}

class OldMultinomialLogisticTest
  extends PerfTest[OldLogisticRegressionModel, RDD[OldLabeledPoint]] {

  def convertData(spark: SparkSession, rdd: RDD[LabeledPoint]): RDD[OldLabeledPoint] = {
    rdd.map {lp => OldLabeledPoint(lp.label, OldVectors.dense(lp.features.toArray))}
  }

  def runTest(
      data: RDD[OldLabeledPoint],
      args: scala.collection.mutable.HashMap[String, String],
      bestOf: Int): (OldLogisticRegressionModel, Double) = {
    val regParam = args.getOrElse("regParam", "0.0").toDouble
    val elasticNetParam = args.getOrElse("elasticNetParam", "0.0").toDouble
    val standardization = args.getOrElse("standardization", "true").toBoolean
    val fitIntercept = args.getOrElse("fitIntercept", "true").toBoolean
    val numClasses = args.getOrElse("numClasses", "3").toInt
    val maxIter = args.getOrElse("maxIter", "100").toInt
    val algo = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)
      .setIntercept(fitIntercept)
    val optimizer = algo.optimizer
    optimizer.setRegParam(regParam)
    optimizer.setNumIterations(maxIter)


    val results = (0 until bestOf).map { i =>
      val t0 = System.currentTimeMillis()
      val model = algo.run(data)
      val t1 = System.currentTimeMillis()
      (model, (t1 - t0) / 1e3)
    }
    results.minBy(_._2)
  }

  def validate(model: OldLogisticRegressionModel, testData: RDD[OldLabeledPoint]): Double = {
    val correct = testData.map { lp =>
      if (model.predict(lp.features) == lp.label) 1 else 0
    }.sum()
    correct / testData.count.toDouble
  }
}

class LogisticTest extends PerfTest[LogisticRegressionModel, DataFrame] {

  def convertData(spark: SparkSession, rdd: RDD[LabeledPoint]): DataFrame = {
    spark.createDataFrame(rdd)
  }

  def runTest(
               data: DataFrame,
               args: scala.collection.mutable.HashMap[String, String],
               bestOf: Int): (LogisticRegressionModel, Double) = {
    val regParam = args.getOrElse("regParam", "0.0").toDouble
    val elasticNetParam = args.getOrElse("elasticNetParam", "0.0").toDouble
    val standardization = args.getOrElse("standardization", "true").toBoolean
    val fitIntercept = args.getOrElse("fitIntercept", "true").toBoolean
    val maxIter = args.getOrElse("maxIter", "100").toInt
    val lr = new LogisticRegression()
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setFitIntercept(fitIntercept)
      .setStandardization(standardization)
      .setMaxIter(maxIter)

    val results = (0 until bestOf).map { i =>
      val t0 = System.currentTimeMillis()
      val model = lr.fit(data)
      val t1 = System.currentTimeMillis()
      (model, (t1 - t0) / 1e3)
    }
    results.minBy(_._2)
  }

  def validate(model: LogisticRegressionModel, testData: DataFrame): Double = {
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    evaluator.evaluate(model.transform(testData))
  }
}
