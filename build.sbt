name := "MultinomialLogisticPerfTest"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.0-SNAPSHOT",
  "org.apache.spark" %% "spark-mllib" % "2.1.0-SNAPSHOT",
  "org.apache.spark" %% "spark-mllib-local" % "2.1.0-SNAPSHOT",
  "org.apache.spark" %% "spark-sql" % "2.1.0-SNAPSHOT",
  "au.com.bytecode" % "opencsv" % "2.4",
  "com.github.tototoshi" %% "scala-csv" % "1.3.1"
)
    