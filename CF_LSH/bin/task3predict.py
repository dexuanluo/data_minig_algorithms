import sys
from pyspark import SparkConf
from pyspark.context import SparkContext
import json
from collections import Counter
from itertools import combinations


USER_BASED = "user_based"
ITEM_BASED = "item_based"
USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
STAR = "stars"
B1_TAG = "b1"
B2_TAG = "b2"
U1_TAG = "u1"
U2_TAG = "u2"
SIM_TAG = "sim"
MAX_NEIG = 5

def make_init_tuple(x):
    return (x[BUSINESS_ID],x[USER_ID], x[STAR])

def make_test_tuple(x):
    return (x[USER_ID], x[BUSINESS_ID])

def reshape_model_item(x):
    return [(x[B1_TAG],(x[SIM_TAG], x[B2_TAG])), (x[B2_TAG],(x[SIM_TAG], x[B1_TAG]))]

def reshape_model_user(x):
    return [(x[U1_TAG],(x[SIM_TAG], x[U2_TAG])), (x[U2_TAG],(x[SIM_TAG], x[U1_TAG]))]

def item_based_predict(x, up, bp, model):
    user, business = x
    candidate = []
    for pearson, neig in model[business]:
        if neig in up[user]:
            candidate.append((pearson, bp[neig], neig))

    candidate.sort(reverse = True)
    # candidate = candidate[:MAX_NEIG]
    r1, r2 = 0, 0
    for pearson, neig_avg, neig in candidate:
        r1 += (up[user][neig] - neig_avg) * pearson
        r2 += abs(pearson)

    return (user, business, bp[business] + r1 / (r2 + 0.0000000001))

def user_based_predict(x, up, bp, model):
    user, business = x
    candidate = []
    for pearson, neig in model[user]:
        if neig in bp[business]:
            candidate.append((pearson, up[neig], neig))
    candidate.sort(reverse = True)
    candidate = candidate[:MAX_NEIG]
    r1, r2 = 0, 0
    for pearson, neig_avg, neig in candidate:
        r1 += (bp[business][neig] - neig_avg) * pearson
        r2 += abs(pearson)
    return (user, business, up[user] + r1 / (r2 + 0.0000000001))

if __name__ == "__main__":
    argv = sys.argv
    train_path = argv[1]
    test_path = argv[2]
    model_path = argv[3]
    output_path = argv[4]
    cf_type = argv[5]

    conf = SparkConf().setAppName("task3predict")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    
    sc = SparkContext(conf = conf)

    js_data = sc.textFile(train_path)

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

    js_data = js_data.map(lambda x : (business_map[x[0]], users_map[x[1]], x[2]))

    user_profile = js_data\
        .map(lambda x : (x[1], (x[0], x[2])))\
            .groupByKey()\
                .map(lambda x : (x[0], {b_id: rating for b_id, rating in x[1]}))
                    

    business_profile = js_data\
        .map(lambda x : (x[0], (x[1], x[2])))\
            .groupByKey()\
                .map(lambda x : (x[0], {u_id: rating for u_id, rating in x[1]}))
                    
    
    model_data = sc.textFile(model_path)

    test_data = sc.textFile(test_path)

    test_data = test_data.map(lambda x : make_test_tuple(json.loads(x)))

    if cf_type == ITEM_BASED:

        user_profile = user_profile.collectAsMap()

        business_profile = business_profile\
            .map(lambda x : (x[0], sum([x[1][key] for key in x[1]]) / len(x[1])))\
                .collectAsMap()

        model_data = model_data\
            .flatMap(lambda x : reshape_model_item(json.loads(x)))\
                .map(lambda x : (business_map[x[0]], (x[1][0], business_map[x[1][1]])))\
                    .groupByKey()\
                        .map(lambda x : (x[0], tuple(x[1])))\
                            .collectAsMap()
        
        test_data = test_data\
            .filter(lambda x : x[0] in users_map and x[1] in business_map)\
                .map(lambda x : (users_map[x[0]], business_map[x[1]]))\
                    .filter(lambda x : x[0] in user_profile and x[1] in model_data)
                

        result = test_data\
            .map(lambda x : item_based_predict(x, user_profile, business_profile, model_data))\
                .collect()

    if cf_type == USER_BASED:

        business_profile = business_profile.collectAsMap()

        user_profile = user_profile\
            .map(lambda x : (x[0], sum([x[1][key] for key in x[1]]) / len(x[1])))\
                .collectAsMap()
        
        model_data = model_data\
            .flatMap(lambda x : reshape_model_user(json.loads(x)))\
                .map(lambda x : (users_map[x[0]], (x[1][0], users_map[x[1][1]])))\
                    .groupByKey()\
                        .map(lambda x : (x[0], tuple(x[1])))\
                            .collectAsMap()
        
        test_data = test_data\
            .filter(lambda x : x[0] in users_map and x[1] in business_map)\
                .map(lambda x : (users_map[x[0]], business_map[x[1]]))\
                    .filter(lambda x : x[0] in model_data and x[1] in business_profile)
                

        result = test_data\
            .map(lambda x : user_based_predict(x, user_profile, business_profile, model_data))\
                .collect()


    with open(output_path, "w") as f:
        for val1, val2, rating in result:
            f.write(json.dumps({USER_ID:all_users[val1], BUSINESS_ID:all_business[val2], STAR:rating}) + "\n")
        