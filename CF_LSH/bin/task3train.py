import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
import json
from collections import Counter
from itertools import combinations
from math import log2
from random import randint

ITEM_BASED = "item_based"
USER_BASED = "user_based"
USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
STAR = "stars"
B1_TAG = "b1"
B2_TAG = "b2"
U1_TAG = "u1"
U2_TAG = "u2"
SIM_TAG = "sim"
MIN_CORATED = 3
ITEM_THRESHOLD = 0

JACCARD_THRESHOLD = 0.01
NUM_HASH_FUNC = 100
NUM_BAND = 100
NUM_ROW = NUM_HASH_FUNC // NUM_BAND

MAGIC_NUMBER = 66633333111223333

def get_hash_param(num_of_hash):
    n = 0
    hash_param = set()
    while n < num_of_hash:
        a, b = randint(1, 10000000), randint(0, 10000000)
        while (a, b) in hash_param:
            a, b = randint(1, 10000000), randint(0, 10000000)
        hash_param.add((a, b))
        n += 1
    return list(hash_param)

def make_init_tuple(x):
    return (x[BUSINESS_ID],x[USER_ID], x[STAR])

def permutation_hash(x, a, b, m):
    return ((a*x + b) % MAGIC_NUMBER) % m

def min_hash(x, max_idx, hash_param):
    key_idx, val_arr = x
    signature = [float("Inf")] * len(hash_param)
    for i in range(len(hash_param)):
        a, b = hash_param[i]
        for val in val_arr:
            signature[i] = min(signature[i], permutation_hash(val, a, b, max_idx))
    return (key_idx, tuple(signature))

def lsh(x, r, b):
    buckets = []
    key_id, sig = x
    for i in range(b):
        buckets.append(((i, hash(sig[i * r: i * r + r])), key_id))
    return buckets

def jaccard_sim(a, b, d):
    n1 = len(set(d[a]) & set(d[b]))
    if n1 >= 3:
        n2 = len(set(d[b]) | set(d[b]))
        return n1 / n2 if n2 > 0 else 0
    return 0

def pearson(b1, b2):
    b1_bid, b1_ratings = b1
    b2_bid, b2_ratings = b2
    
    s2 = {user : rating for user, rating in  b2_ratings}
    r1, r2, = [], []
    for user, rating in b1_ratings:
        if user in s2:
            r1.append(rating)
            r2.append(s2[user])  
    co_rated = len(r1)
    r1_avg = sum(r1) / len(r1)
    r2_avg = sum(r2) / len(r2)
    r = sum([(r1[i] - r1_avg) * (r2[i] - r2_avg) for i in range(co_rated)])
    r1q = sum([(r1[i] - r1_avg)**2 for i in range(co_rated)])
    r2q = sum([(r2[i] - r2_avg)**2 for i in range(co_rated)])
    pearson = r / ((r1q**0.5) * (r2q**0.5) + 0.00000000001)
    return (b1_bid, b2_bid, pearson)

if __name__ == "__main__":
    argv = sys.argv
    input_path = argv[1]
    output_path = argv[2]
    cf_type = argv[3]

    conf = SparkConf().setAppName("task3train")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    
    sc = SparkContext(conf = conf)

    js_data = sc.textFile(input_path)

    js_data = js_data\
        .map(lambda x : make_init_tuple(json.loads(x)))

    all_business = js_data\
        .map(lambda x : x[0])\
            .distinct()\
                .collect()

    all_users = js_data\
        .map(lambda x : x[1])\
            .distinct()\
                .collect()

    all_business.sort()
    business_map = {all_business[i] : i for i in range(len(all_business))}

    all_users.sort()
    users_map = {all_users[i] : i for i in range(len(all_users))}

    num_user = len(all_users)
    num_business = len(all_business)

    if cf_type == ITEM_BASED:

        business_profile = js_data\
            .map(lambda x : (business_map[x[0]], (users_map[x[1]], x[2])))\
                .groupByKey()\
                    .map(lambda x : (x[0], tuple(x[1])))\
                        .collect()
        
        business_profile.sort()

        item_based_model = []

        pairs = combinations(business_profile, 2)
        
        for b1, b2 in pairs:
            
            b1_bid, b1_ratings = b1
            b2_bid, b2_ratings = b2
            
            s2 = {user : rating for user, rating in  b2_ratings}

            if len(b1_ratings) >= MIN_CORATED and len(b2_ratings) >= MIN_CORATED:
                
                r1, r2, = [], []
                for user, rating in b1_ratings:
                    if user in s2:
                        r1.append(rating)
                        r2.append(s2[user]) 
                co_rated = len(r1)  
                if co_rated >= MIN_CORATED:
                    r1_avg = sum(r1) / co_rated
                    r2_avg = sum(r2) / co_rated
                    r = sum([(r1[i] - r1_avg) * (r2[i] - r2_avg) for i in range(co_rated)])
                    r1q = sum([(r1[i] - r1_avg)**2 for i in range(co_rated)])
                    r2q = sum([(r2[i] - r2_avg)**2 for i in range(co_rated)])
                    pearson = r / ((r1q**0.5) * (r2q**0.5) + 0.00000000001)
                    if pearson > ITEM_THRESHOLD:
                        item_based_model.append((b1_bid, b2_bid, pearson))
                        
        item_based_model.sort(reverse = True)

    if cf_type == USER_BASED:

        hash_param = get_hash_param(NUM_HASH_FUNC)

        js_data = js_data\
            .map(lambda x : (users_map[x[1]], (business_map[x[0]], x[2])))
        
        reserved_data = js_data\
            .map(lambda x : (x[0], x[1][0]))\
                .groupByKey()\
                    .map(lambda x : (x[0], tuple(x[1])))\

        signature_matrix = reserved_data\
            .map(lambda x : min_hash(x, num_business * 2, hash_param))
        
        lsh_buckets = signature_matrix\
            .flatMap(lambda x : lsh(x, NUM_ROW, NUM_BAND))\
                .groupByKey()\
                    .map(lambda x : tuple(x[1]))\
                        .filter(lambda x : len(x) > 1)\
                            .flatMap(lambda x : [pairs for pairs in combinations(x, 2)])\
                                .map(lambda x : tuple(sorted(x)))\
                                    .distinct()

        set_dict = reserved_data.collectAsMap()
        
        freq_pairs = lsh_buckets\
            .map(lambda x : (x, jaccard_sim(x[0], x[1], set_dict)))\
                .filter(lambda x : x[1] >= JACCARD_THRESHOLD)\
                    .map(lambda x : x[0])
        
        user_profile = js_data\
            .groupByKey()\
                .map(lambda x : (x[0], tuple(x[1])))\
                    .collect()

        user_profile.sort()

        user_based_model = freq_pairs\
            .map(lambda x : pearson(user_profile[x[0]], user_profile[x[1]]))\
                .filter(lambda x : x[2] > 0)\
                    .collect()

        user_based_model.sort()

    with open(output_path, "w") as f:
        if cf_type == ITEM_BASED:
            while item_based_model:
                b1, b2, sim = item_based_model.pop()
                f.write(json.dumps({B1_TAG:all_business[b1], B2_TAG:all_business[b2], SIM_TAG: sim}) + "\n")
        elif cf_type == USER_BASED:
            while user_based_model:
                u1, u2, sim = user_based_model.pop()
                f.write(json.dumps({U1_TAG:all_users[u1], U2_TAG:all_users[u2], SIM_TAG: sim}) + "\n")
        else:
            print("Wrong CF type")
        

        