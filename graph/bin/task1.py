import sys
import os
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import *
from itertools import combinations
from graphframes import *

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == "__main__":
    argv = sys.argv
    THRESHOLD = int(argv[1])
    input_path = argv[2]
    output_path = argv[3]
    
    conf = SparkConf().setAppName("task1")\
        .setMaster("local[3]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")\

                    

    sc = SparkContext(conf = conf)

    sc.setLogLevel("OFF")

    sqlc = SQLContext(sc)

    raw_data = sc.textFile(input_path)

    raw_data = raw_data\
        .map(lambda x : tuple(x.split(",")))\
            .filter(lambda x : x[0] != "user_id")

    b2u = raw_data\
        .map(lambda x : (x[1], x[0]))\
            .groupByKey()\
                .map(lambda x : (x[0], sorted(tuple(x[1]))))
    
    edges = b2u\
        .flatMap(lambda x : [(pair, 1) for pair in combinations(x[1], 2)])\
            .reduceByKey(lambda x, y : x + y)\
                    .filter(lambda x : x[1] >= THRESHOLD)\
                        .flatMap(lambda x : [x[0], (x[0][1], x[0][0])])
                            

    vertices = edges\
        .map(lambda x : (x[0],))\
            .distinct()
    

    
    vertices_set = sqlc.createDataFrame(vertices, ["id"])

    edges_set = sqlc.createDataFrame(edges, ["src", "dst"])

    graph = GraphFrame(vertices_set, edges_set)

    result = graph.labelPropagation(maxIter=5)
    
    result = result\
        .rdd\
            .map(tuple)\
                .map(lambda x : (x[1], x[0]))\
                    .groupByKey()\
                        .map(lambda x : (x[0], sorted(tuple(x[1]))))\
                            .map(lambda x : x[1])\
                                .collect()

    result.sort(key = lambda x : (len(x), x))
    
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = "'" + result[i][j] + "'"

    with open(output_path, "w") as f:
        for com in result:
            f.write(", ".join(com) + "\n")
    

    
    
    


    



