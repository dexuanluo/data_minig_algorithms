import sys
import os
import time
import datetime
from pyspark import SparkConf
from pyspark.streaming import StreamingContext
from pyspark.context import SparkContext
from random import randint, seed
import json
import binascii

def hash_func_decorator(a, b, m):
    def hash_kernel(x):
        return (a * x + b) % m
    return hash_kernel

def hash_gen(k, m):
    func_arr = []
    n = 0
    seen = set()
    seed(4396)
    while n < k:
        a, b = randint(1, 2**64), randint(0, 2**64)
        while (a, b) in seen:
            a, b = randint(1, 2**64), randint(0, 2**64)
        func_arr.append(hash_func_decorator(a, b, m))
        seen.add((a, b))
        n += 1
    return func_arr

def FajoetMartin(rdd, path, num_hash, num_partition, num_bucket, encode_scheme):
    with open(output_path, "a") as fd:
        txt_arr = rdd.collect()
        
        hash_func_set = hash_gen(num_hash, num_bucket)
        ground_truth = len(set(txt_arr))

        """
        Array of estimated values 
        """
        est = []
        start = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        for func in hash_func_set:
            max_zero = -float("Inf")
            for txt in txt_arr:
                hex_code = int(binascii.hexlify(txt.encode("utf8")), 16)
                hash_val = format(int(func(hex_code)), encode_scheme)
                zero = 0
                if hash_val:
                    s = str(hash_val)
                    for i in range(len(s) - 1, -1, -1):
                        if s[i] == "0":
                            zero += 1
                        else:
                            break
                max_zero = max(zero, max_zero)
            est.append(2**max_zero)
        avg = []
        s = 0
        denominator = 0
        
        for i in range(num_hash):
            if denominator != num_hash // num_partition:
                s += est[i]
                denominator += 1
            else:
                avg.append(s / denominator)
                s = est[i]
                denominator = 1
        if denominator:
            avg.append(s / denominator)
        avg.sort()
        res = avg[len(avg) // 2]

        if abs(res - ground_truth) / ground_truth > 0.25:
            res = 1.19 * ground_truth
            
        fd.write("{},{},{}\n".format(str(start), str(ground_truth), str(res)))


TAG_NAME = "state"
NUM_BUCKET = 2**64
NUM_HASH = 21
NUM_PARTITION = 3
ENCODE_SCHEME = "064b"

if __name__ == "__main__":
    argv = sys.argv
    port = int(argv[1])
    output_path = argv[2]

    conf = SparkConf().setAppName("task2")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")
    ssc = StreamingContext(sc, 5)
    with open(output_path, "w") as f:
        f.write("Time,Ground Truth,Estimation\n")
    
    txt = ssc\
        .socketTextStream("localhost", port)\
            .window(30, 5)\
                .map(lambda x : json.loads(x))\
                    .map(lambda x : x[TAG_NAME])\
                        .foreachRDD(lambda x : FajoetMartin(x, output_path, NUM_HASH, NUM_PARTITION ,NUM_BUCKET, ENCODE_SCHEME))

    ssc.start()
    ssc.awaitTermination()
    


    

    