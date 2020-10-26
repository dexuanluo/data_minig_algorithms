import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
import json
from random import randint
from itertools import combinations

def jaccard_sim(a, b, d):
    n1 = len(d[a] & d[b])
    n2 = len(d[a] | d[b])
    return n1 / n2 if n2 > 0 else 0
    
def get_bus_user_tuple(js):
    return (js["business_id"], js["user_id"])
def make_set(x, d):
    return (x[0], d[x[1]])

def permutation_hash(x, a, b, m):
    return ((a*x + b) % 66633333111223333) % m

def min_hash(x, max_idx, hash_param):
    business_idx, users_arr = x
    signature = [float("Inf")] * len(hash_param)
    for i in range(len(hash_param)):
        a, b = hash_param[i]
        for user in users_arr:
            signature[i] = min(signature[i], permutation_hash(user, a, b, max_idx))
    return (business_idx, tuple(signature))

def lsh(x, r, b):
    buckets = []
    b_id, sig = x
    for i in range(b):
        buckets.append(((i, hash(sig[i * r: i * r + r])), b_id))
    
    return buckets

def reduce_bucket(x, y): 
    res = set(x)
    for business in y:
        res.add(business)
    return tuple(sorted(list(res)))


THRESHOLD = 0.055
NUM_HASH_FUNC = 100
num_band = 100     

if __name__ == "__main__":
    argv = sys.argv
    input_path = argv[1]
    output_path = argv[2]
    
    conf = SparkConf().setAppName("task1")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")

    sc = SparkContext(conf = conf)

    num_r = NUM_HASH_FUNC // num_band
    n = 0
    hash_param = set()
    while n < NUM_HASH_FUNC:
        a, b = randint(1, 10000000), randint(0, 10000000)
        while (a, b) in hash_param:
            a, b = randint(1, 10000000), randint(0, 10000000)
        hash_param.add((a, b))
        n += 1
    hash_param = list(hash_param)

    js_data = sc.textFile(input_path)

    js_data = js_data.map(lambda x : get_bus_user_tuple(json.loads(x)))

    all_users = js_data.map(lambda x : (x[1], 1))\
        .reduceByKey(lambda x, y : 1)\
            .collect()

    all_business = js_data.map(lambda x : (x[0], 1))\
        .reduceByKey(lambda x, y : 1)\
            .collect()

    all_users.sort()
    all_business.sort()
    num_users, num_business = len(all_users), len(all_business)

    users_map = {}
    business_map = {}
    for i in range(num_business):
        all_business[i] = all_business[i][0]
    for i in range(num_users):
        all_users[i] = all_users[i][0]
    for i in range(num_users):
        users_map[all_users[i]] = i
    for i in range(num_business):
        business_map[all_business[i]] = i
    
    
    js_data = js_data\
        .map(lambda x : (business_map[x[0]], users_map[x[1]]))\
            .groupByKey()\
                .map(lambda x : (x[0], tuple(x[1])))
    
    signature_matrix =js_data.map(lambda x : min_hash(x, num_users * 2, hash_param))

    lsh_buckets = signature_matrix.flatMap(lambda x : lsh(x, num_r, num_band))

    lsh_buckets = lsh_buckets.groupByKey()\
        .map(lambda x : tuple(x[1]))\
            .filter(lambda x : len(x) > 1)\
                .flatMap(lambda x : [pairs for pairs in combinations(x, 2)])\
                    .map(lambda x : tuple(sorted(x)))\
                        .distinct()

    b2u_dict = js_data\
        .map(lambda x : (x[0], set(x[1])))\
            .collectAsMap()
    
    lsh_buckets = lsh_buckets\
        .map(lambda x : (x, jaccard_sim(x[0], x[1], b2u_dict)))\
            .filter(lambda x : x[1] >= THRESHOLD)

    freq_pairs = lsh_buckets.collect()

    freq_pairs.sort(reverse = True)

    with open(output_path, "w") as f:
        while freq_pairs:
            bus, sim = freq_pairs.pop()
            b1, b2 = bus
            f.write("{"+ "\"b1\":\"{}\", \"b2\":\"{}\", \"sim\":{}".format(all_business[b1], all_business[b2], sim)+"}\n")
    
    
    

    

    
    
        


        
