import sys
import os
from pyspark import SparkConf
from pyspark.context import SparkContext
from random import randint
import json
import binascii

def hash_func_decorator(a, b, m):
    def hash_kernel(x):
        return ((a * x + b) % 71371371371731731) % m
    return hash_kernel

def hash_gen(k, m):
    func_arr = []
    n = 0
    seen = set()
    while n < k:
        a, b = randint(1, sys.maxsize - 1), randint(0, sys.maxsize - 1)
        while (a, b) in seen:
            a, b = randint(1, sys.maxsize - 1), randint(0, sys.maxsize - 1)
        func_arr.append(hash_func_decorator(a, b, m))
        seen.add((a, b))
        n += 1
    return func_arr
    
class BloomFilter:
    def __init__(self, hash_num, bucket_num):
        self.func = hash_gen(hash_num, bucket_num)
        self.set_table = [set() for _ in range(hash_num)]
    def add(self, row, pos):
        self.set_table[row].add(pos)
    def is_set(self, row, pos):
        return pos in self.set_table[row]
    def train(self, num):
        for i, func in enumerate(self.func):
            self.add(i, func(num))
    def predict(self, num):
        for i, func in enumerate(self.func):
            if not self.is_set(i, func(num)):
                return False
        return True

NUM_BCKT = 10000000
NUM_HASH = 100
TAG_NAME = "name"

if __name__ == "__main__":
    argv = sys.argv
    input_path1 = argv[1]
    input_path2 = argv[2]
    output_path = argv[3]

    conf = SparkConf().setAppName("task1")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")

    sc = SparkContext(conf = conf)

    bf = BloomFilter(NUM_HASH, NUM_BCKT)
    b1 = sc.textFile(input_path1)

    b1 = b1\
        .map(lambda x : json.loads(x))\
            .map(lambda x : x[TAG_NAME])\
                .map(lambda s : int(binascii.hexlify(s.encode('utf8')),16))\
                    .collect()
    
    while b1:
        num = b1.pop()
        bf.train(num)
    
    res = []
    
    with open(input_path2, "r") as f:
        line = f.readline()
        while line:
            s = json.loads(line)[TAG_NAME]
            business = int(binascii.hexlify(s.encode('utf8')),16)
            if bf.predict(business):
                res.append("T")
            else:
                res.append("F")
            line = f.readline()

    with open(output_path, "w") as f:
        f.write(" ".join(res))


    
    

    

    





