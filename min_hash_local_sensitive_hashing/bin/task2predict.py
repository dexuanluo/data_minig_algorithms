import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
import json
from collections import Counter


USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
BREAK_POINT = "##break##"
THRESHOLD = 0.01

def get_test_tuple(x):
    js = json.loads(x)
    return (js[USER_ID], js[BUSINESS_ID])
def cosine_sim(x, up, bp, all_bus):
    user, business = x
    if user in up and business in bp:
        u_s = set()
        for b_id in up[user]:
            u_s |= set(bp[all_bus[b_id]])
        u = len(u_s & set(bp[business]))
        return ((user, business), u / (len(u_s)**0.5 * len(bp[business])**0.5))
    return ((0, 0), 0)
if __name__ == "__main__":
    argv = sys.argv
    input_path = argv[1]
    model_path = argv[2]
    output_path = argv[3]
    
    conf = SparkConf().setAppName("task2predict")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    
    sc = SparkContext(conf = conf)

    users_profile = {}
    business_profile = {}

    with open(model_path, "r") as f:
        line = f.readline()
        line = line.rstrip()
        while line != BREAK_POINT:
            js = json.loads(line)
            for key in js:
                users_profile[key] = {}
                users_profile[key] = tuple(js[key])
                
            line = f.readline()
            line = line.rstrip()
        
        line = f.readline()

        while line:
            js = json.loads(line)
            for key in js:
                business_profile[key] = tuple(js[key])
            line = f.readline()
            line = line.rstrip()

    all_business = []

    for bus in business_profile:
        all_business.append(bus)

    all_business.sort()

    test_set = sc.textFile(input_path)

    result = test_set.map(lambda x : get_test_tuple(x))\
        .map(lambda x :cosine_sim(x, users_profile, business_profile, all_business))\
            .filter(lambda x : x[1] >= THRESHOLD)\
                .collect()

    result.sort(reverse = True)
    
    with open(output_path, "w") as f:
        while result:
            t, sim = result.pop()
            user, business = t
            f.write(json.dumps({USER_ID: user, BUSINESS_ID: business, "sim": sim}) + "\n")
    
    

    


