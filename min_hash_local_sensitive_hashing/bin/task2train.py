import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
import json
from collections import Counter
from math import log2

USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
TEXT = 'text'
MAX_FEATURES = 200

def clean_text(x):
    punc = "(@~|[,.!?:;])/\\*\"'&+-%#1234567890$^"
    x = x.lower()
    for p in punc:
        x = x.replace(p, "")
    
    return x
    
def make_init_tuple(x):
    return (x[BUSINESS_ID], x[USER_ID], clean_text(x[TEXT]))

def text2bus(x, stopwords, rare_words):
    business, _, txt = x
    counter = Counter(txt.split())
    res = []
    for word in counter:
        if word not in stopwords and word not in rare_words:
            res.append((word, (counter[word], business)))
    return res
    
def get_word_count(x, stopwords):
    _, _, txt = x
    res = []
    counter = Counter(txt.split())
    for word in counter:
        if word not in stopwords:
            res.append((word, counter[word]))
    return res

def calculate_inverse(x):
    txt, b_arr = x
    counter = {}
    for count, b_id in b_arr:
        if b_id not in counter:
            counter[b_id] = 0
        counter[b_id] += count
    n = len(counter)
    res = []
    for business in counter:
        res.append((business, (counter[business], hash(txt), n)))
    return res

def calculate_tfidf(x, num_of_all_bus, feature_size):
    business, word_arr = x
    max_freq = word_arr[0][0]
    tfidf = []
    while word_arr:
        count, txt, doc_mentioned = word_arr.pop()
        tfidf.append((log2(num_of_all_bus / doc_mentioned) * count / max_freq, txt))
    tfidf.sort(reverse = True)
    return (business, [txt for _, txt in tfidf[:feature_size]])


if __name__ == "__main__":
    argv = sys.argv
    input_path = argv[1]
    output_path = argv[2]
    stopwords_path = argv[3]
    
    conf = SparkConf().setAppName("task2train")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")

    sc = SparkContext(conf = conf)

    stopwords = set()

    with open(stopwords_path, "r") as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            stopwords.add(line)
            line = f.readline()
    
    js_data = sc.textFile(input_path)

    js_data = js_data.map(lambda x : make_init_tuple(json.loads(x)))

    all_business = js_data.map(lambda x : x[0]).distinct().collect()
    all_business.sort()
    
    business_map = {}
    for i in range(len(all_business)):
        business_map[all_business[i]] = i

    NUM_OF_BUSINESS = len(all_business)

    js_data = js_data.map(lambda x : (business_map[x[0]], x[1], x[2]))

    #35 is 0.0001% of the total number of words
    rare_words = js_data.flatMap(lambda x : get_word_count(x, stopwords))\
        .reduceByKey(lambda x, y: x + y)\
            .filter(lambda x: x[1] <= 35)\
                .map(lambda x : x[0])\
                    .collect()
            

    rare_words = set(rare_words)
           
    tfidf = js_data.flatMap(lambda x : text2bus(x, stopwords, rare_words))\
        .groupByKey()\
            .map(lambda x : (x[0], tuple(x[1])))\
                .flatMap(lambda x : calculate_inverse(x))\
                    .groupByKey()\
                        .map(lambda x :(x[0], sorted(list(x[1]), reverse = True)))\
                            .map(lambda x: calculate_tfidf(x, NUM_OF_BUSINESS, MAX_FEATURES))

    business_profile = tfidf.map(lambda x : (x[0], set(x[1]))).collectAsMap()
    
    user_profile = js_data.map(lambda x : (x[1], x[0]))\
        .groupByKey()\
            .map(lambda x : (x[0], tuple(x[1])))\
                .collectAsMap()
    
    
    with open(output_path, "w") as f:
        for user in user_profile:
            if len(user_profile[user]) > 2:
                f.write((json.dumps({user : list(user_profile[user])}) + "\n"))

        f.write("##break##\n")

        for b_id in business_profile:
            f.write(json.dumps({all_business[b_id] : list(business_profile[b_id])}) + "\n")
    
    
        