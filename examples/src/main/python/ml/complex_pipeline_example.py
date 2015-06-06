#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.feature import HashingTF, StringIndexer, Tokenizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import HiveContext, Row


"""
An example of an end to end machine learning pipeline that classifies text
into one of twenty possible news categories. The dataset is the 20newsgroups
dataset (http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)

We assume some minimal preprocessing of this dataset has been done to unzip the dataset and
load the data into HDFS as follows:
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -xvzf 20news-bydate.tar.gz
hadoop fs -mkdir ${20news.root.dir}
hadoop fs -copyFromLocal 20news-bydate-train/ ${20news.root.dir}
hadoop fs -copyFromLocal 20news-bydate-test/ ${20news.root.dir}

This example uses Hive to schematize the data as tables, in order to map the folder
structure ${20news.root.dir}/{20news-bydate-train, 20news-bydate-train}/{newsgroup}/
to partition columns type, newsgroup resulting in a dataset with three columns:
type, newsgroup, text

In order to run this example, Spark needs to be build with hive, and at runtime there
should be a valid hive-site.xml in $SPARK_HOME/conf with at minimal the following
configuration:
<configuration>
    <property>
        <name>hive.metastore.uris</name>
    <!-- Ensure that the following statement points to the Hive Metastore URI in your cluster -->
        <value>thrift://${thriftserver.host}:${thriftserver.port}</value>
        <description>URI for client to contact metastore server</description>
    </property>
</configuration>

Run with
{{{
    bin/spark-submit --class org.apache.spark.examples.ml.ComplexPipelineExample
      --driver-memory 4g [examples JAR path] ${20news.root.dir}
}}}
"""

if __name__ == "__main__":
    sc = SparkContext(appName="ComplexPipelineExample")
    sqlContext = HiveContext(sc)

    # Set up Hive External Table for processing 20Newsgroups dataset
    sqlContext.sql("""CREATE EXTERNAL TABLE IF NOT EXISTS 20NEWS(text String)
      PARTITIONED BY (type String, newsgroup String)
      STORED AS TEXTFILE location '/20newsgroups'""")

    newsgroups = ["alt.atheism", "comp.graphics",
                  "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                  "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale",
                  "rec.autos", "rec.motorcycles", "rec.sport.baseball",
                  "rec.sport.hockey", "sci.crypt", "sci.electronics",
                  "sci.med", "sci.space", "soc.religion.christian",
                  "talk.politics.guns", "talk.politics.mideast",
                  "talk.politics.misc", "talk.religion.misc"]

    for t in ["20news-bydate-train", "20news-bydate-test"]:
        for newsgroup in newsgroups:
            sqlContext.sql("""ALTER TABLE 20NEWS ADD IF NOT EXISTS
              PARTITION(type='%s', newsgroup='%s')
              LOCATION '/20newsgroups/%s/%s/'""" % (t, newsgroup, t, newsgroup))

    # shuffle the data
    # by default we have over 19k partitions
    partitions = 100
    data = sqlContext.sql("SELECT * FROM 20NEWS") \
        .coalesce(partitions) \
        .repartition(partitions) \
        .cache()

    train = data.filter(data.type == "20news-bydate-train").cache()
    test = data.filter(data.type == "20news-bydate-test").cache()

    # convert string labels into numeric
    labelIndexer = StringIndexer(inputCol = "newsgroup", outputCol = "label")

    # tokenize text into words
    tokenizer = Tokenizer(inputCol = "text", outputCol = "words")

    # extract hash based TF-IDF features
    hashingTF = HashingTF(numFeatures = 1000,
                          inputCol = tokenizer.getOutputCol,
                          outputCol = "features")

    # learn multiclass classifier with Logistic Regression as base classifier
    lr = LogisticRegression(maxIter = 10)

    ovr = OneVsRest(classifier = lr)

    pipeline = Pipeline(stages = [labelIndexer, tokenizer, hashingTF, ovr])

    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10, 100, 1000, 2000, 5000]) \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .build()
    # cross validate
    cv = CrossValidator(estimator = pipeline,
                        evaluator = MulticlassClassificationEvaluator,
                        estimatorParamMaps = paramGrid,
                        numFolds = 3)

    # select best model
    model = cv.fit(train)

    # score the model
    predictions = model.transform(test).cache()

    predictionAndLabels = predictions.select("prediction", "label") \
        .map(lambda x: (x.prediction, x.label))

    # compute multiclass metrics
    metrics = MulticlassMetrics(predictionAndLabels)

    labelToIndexMap = predictions.select("label", "newsgroup") \
        .distinct() \
        .map(lambda s: (s.newsgroup, s.label)) \
        .collectAsMap()

    for newsgroup, label in labelToIndexMap.iteritems():
        print "%s\t%s" % (newsgroup, metrics.falsePositiveRate(label))

    sc.stop()
