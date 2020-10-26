import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
from itertools import combinations
import time

def merge_by_key(x, y):
    res = set()
    
    for node in x:
        res.add(node)
    
    for elem in y:
        res.add(elem)
    return tuple(res)

def preprocess(x):
    x  = x.split(",")
    return (x[0].rstrip(), (x[1].rstrip(),))

def apriori(partition, total, s):
    l = 0
    count = {}
    buckets = []
    
    for _, item in partition:
        l += 1
        for b in item:
            if b not in count:
                count[b] = 0
            count[b] += 1
        buckets.append(set(item))
    
    threshold = (l / total) * s
    res = set()
    for item in count:
        if count[item] >= threshold:
            res.add((item,))
    max_pair_size = 0
    comb_num = 2
    to_remove = []

    for i in range(len(buckets)):
        for item in buckets[i]:
            if (item,) not in res:
                to_remove.append(item)
        while to_remove:
            buckets[i].remove(to_remove.pop())
        buckets[i] = sorted(list(buckets[i]))
        max_pair_size = max(len(buckets[i]), max_pair_size)
    
    while comb_num <= max_pair_size:
        counter = {}
        old_len = len(res)
        for item in buckets:
            combination = combinations(item, comb_num)
            pair = next(combination, None)
            while pair:
                if pair in counter:
                    counter[pair] += 1
                else:
                    flag = True
                    sub_comb_num = 1
                    while sub_comb_num < comb_num:
                        sub_combination = combinations(pair, sub_comb_num)
                        sub_pair = next(sub_combination, None)
                        while sub_pair:
                            if sub_pair not in res:
                                flag = False
                                break
                            sub_pair = next(sub_combination, None)
                        if not flag:
                            break
                        sub_comb_num += 1
                    if flag:
                        counter[pair] = 1
                pair = next(combination, None)
        for pair in counter:
            if counter[pair] >= threshold:
                res.add(pair)
        if len(res) - old_len <= 1:
            break
        comb_num += 1
    return [(pair, 1) for pair in res]

def is_contained(pair1, pair2):
    pair2 = set(pair2)
    for elem in pair1:
        if elem not in pair2:
            return False
    return True

def eliminate_false_positive(partitions, candidates):
    res = []
    for key, item in partitions:
        for candidate, _ in candidates:
            if is_contained(candidate, item):
                res.append((candidate, 1))
    return res

def concat_res(candidates, output_str):
    l = 1
    for i in range(len(candidates)):
        if l != len(candidates[i][0]):
            output_str += "\n\n"
            l = len(candidates[i][0])
        elif l > 1:
            output_str += ","
        
        if l == 1:
            if i < len(candidates) - 1 and len(candidates[i + 1][0]) == l:
                output_str += "('" + str(candidates[i][0][0]) + "'),"
            else:
                output_str += "('" + str(candidates[i][0][0]) + "')"
        else:
            output_str += str(candidates[i][0])
    return output_str

if __name__ == "__main__":

    start_time = int(time.time() * 1000)
    argv = sys.argv
    support = int(argv[1])
    input_path = argv[2]
    output_path = argv[3]
    
    conf = SparkConf().setAppName("task2").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    
    csv_data = sc.textFile(input_path)
    
    csv_data = csv_data.map(lambda x : preprocess(x))

    csv_data = csv_data.filter(lambda x : x[0] != "user_id")
    csv_data = csv_data.reduceByKey(lambda x, y: merge_by_key(x, y))

    total = csv_data.count()

    first_pass = csv_data.mapPartitions(lambda x: apriori(x, total, support)).reduceByKey(lambda x, y: 1)
    
    candidates = first_pass.collect()
    candidates.sort(key = lambda x : (len(x[0]), x))
    second_pass = csv_data.mapPartitions(lambda x : eliminate_false_positive(x, candidates))
    second_pass = second_pass.reduceByKey(lambda x, y: x + y).filter(lambda x : x[1] >= support)
    frequent_items = second_pass.collect()
    frequent_items.sort(key = lambda x : (len(x[0]), x))

    output_str = "Candidates:\n"
    
    output_str = concat_res(candidates, output_str) + "\n\nFrequent Itemsets:\n"

    output_str = concat_res(frequent_items, output_str)
    
    with open(output_path, "w") as f:
        f.write(output_str)

    print('Duration: %.2f' % (int(time.time() * 1000 - start_time) / 1000,))


    


    

    
