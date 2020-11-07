import sys
import os
import json
from collections import Counter
from kmeans import KMeans
from itertools import combinations

class Centroid:
    def __init__(self, dimension):
        self.dimension = dimension
        self.row_sum_avg = [0] * dimension
        self.row_square_sum_avg  = [0] * dimension
        self.N = 0
        self.keys = []

    def get_centroid(self):
        if self.N == 0: return [0] * self.dimension
        return self.row_sum_avg[:]
    
    def get_variance(self):
        return [max(sq - s**2, 0.000000001) for sq, s in zip(self.row_square_sum_avg, self.row_sum_avg)]

    def centroid_update(self, key, point):
        for i in range(len(self.row_sum_avg)):
            self.row_sum_avg[i] = (self.row_sum_avg[i] * self.N + point[i]) / (self.N + 1)
            self.row_square_sum_avg[i] = (self.row_square_sum_avg[i] * self.N + point[i]**2) / (self.N + 1)
        self.N += 1
        self.keys.append(key)
    
    

class Cluster(Centroid):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.points = {}

    def cluster_update(self, key, point):
        self.points[key] = point
        self.centroid_update(key, point)
    
    def is_empty(self):
        return self.N == 0
    
    def pop(self):
        k, res = None, None
        for key in self.points:
            k = key
        res = self.points[key]
        del self.points[key]
        self.N -= 1
        return k, res

def merge_cluster(clusterA, clusterB):
    new_cluster = Cluster(clusterA.dimension)
    for key in clusterA.points:
        new_cluster.cluster_update(key, clusterA.points[key])
    for key in clusterB.points:
        new_cluster.cluster_update(key, clusterB.points[key])
    return new_cluster

class BFR:
    def __init__(self, ncluster, magicnum, idx_arr, matrix, kmeans_tolerance=2, kmeans_max_iter=1000, acceptable_dist_from_centroid = 2):
        self.acceptable_dist_from_centroid = 2
        self.dimension = len(matrix[0])
        self.ncluster = ncluster
        self.magicnum = magicnum
        self.kmeans_tolerance = kmeans_tolerance
        self.kmeans_max_iter = kmeans_max_iter
        self.discard_set = [Centroid(len(matrix[0])) for _ in range(ncluster)]
        self.compression_set = []
        self.retain_set = {}
        self.round = 0
        self.log = ["round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_retained\n"]
        candidate_matrix = []
        can_idx_arr = []
        member = self.kmeans(matrix, self.magicnum)
        count = Counter(member)
        print(count)
        for i in range(len(matrix)):
            if count[member[i]] < 4:
                self.retain_set[idx_arr[i]] = matrix[i]
            else:
                candidate_matrix.append(matrix[i])
                can_idx_arr.append(idx_arr[i])
        
        member = self.kmeans(candidate_matrix, 1)
        for i in range(len(candidate_matrix)):
            self.discard_set[member[i]].centroid_update(idx_arr[i], candidate_matrix[i])
        self.write_log()
        
    def kmeans(self, matrix, offset=1, max_iter=None, tolerance=None, random_init=False, ncluster=None):
        if not max_iter:
            max_iter = self.kmeans_max_iter
        if not tolerance:
            tolerance = self.kmeans_tolerance
        if not ncluster:
            ncluster = self.ncluster
        kmeans = KMeans(matrix, ncluster * offset, random_init=random_init)
        kmeans.kmeans_train(max_iter=max_iter, tolerance=tolerance)
        return kmeans.membership

    def MahalanobisDistance(self, point, centroid, variance):
        return sum([((p - c) / (v**0.5))**2 for p, c, v in zip(point, centroid, variance)])**0.5

    def add_to_discard_set(self, key, point):
        min_md = float("Inf")
        candidate = -1
        for i in range(len(self.discard_set)):
            md = self.MahalanobisDistance(point, self.discard_set[i].get_centroid(), self.discard_set[i].get_variance())
            if md <= self.acceptable_dist_from_centroid * (self.dimension**0.5) and md < min_md:
                candidate = i
                min_md = md
        if candidate >= 0:
            self.discard_set[i].centroid_update(key, point)
            return True
        return False

    def add_to_compresion_set(self, key, point):
        min_md = float("Inf")
        candidate = -1
        for i in range(len(self.compression_set)):
            md = self.MahalanobisDistance(point, self.compression_set[i].get_centroid(), self.compression_set[i].get_variance())
            if md <= self.acceptable_dist_from_centroid * (self.dimension**0.5) and md < min_md:
                candidate = i
                min_md = md
        if candidate >= 0:
            self.compression_set[i].cluster_update(key, point)
            return True
        return False

    def retrain_retain_set(self):
        idx_arr, matrix = [], []
        for key in self.retain_set:
            idx_arr.append(key)
            matrix.append(self.retain_set[key])
            
        member = self.kmeans(matrix, tolerance=0, ncluster=min((self.magicnum * self.ncluster), len(self.retain_set)))
        count = Counter(member)
        cs = {}
        for cluster in count:
            if count[cluster] > 1:
                cs[cluster] = Cluster(len(matrix[0]))
        for i in range(len(matrix)):
            if member[i] in cs:
                del self.retain_set[idx_arr[i]]
                cs[member[i]].cluster_update(idx_arr[i], matrix[i])
        for c in cs:
            self.compression_set.append(cs[c])
    
    def merge_compression_set(self):
        tmp = {}
        for i in range(len(self.compression_set)):
            tmp[i] = self.compression_set[i]
        
        while True:
            candidate = [key for key in tmp]
            d = max(candidate) + 1
            for pair in combinations(candidate, 2):
                key1, key2 = pair
                md1 = self.MahalanobisDistance(tmp[key1].get_centroid(), tmp[key2].get_centroid(), tmp[key2].get_variance())
                md2 = self.MahalanobisDistance(tmp[key2].get_centroid(), tmp[key1].get_centroid(), tmp[key1].get_variance())
                if md1 <= self.acceptable_dist_from_centroid * (self.dimension**0.5) \
                    or md2 <= self.acceptable_dist_from_centroid * (self.dimension**0.5):
                    new_cluster = merge_cluster(tmp[key1], tmp[key2])
                    del tmp[key1]
                    del tmp[key2]
                    tmp[d] = new_cluster
                    d += 1
                    break
            if len(tmp) == len(candidate) or len(tmp) == 1:
                break

        self.compression_set = [tmp[key] for key in tmp]

    def update(self, idx_arr, matrix):
        for key, point in zip(idx_arr, matrix):
            if not self.add_to_discard_set(key, point):
                if not self.add_to_compresion_set(key, point):
                    self.retain_set[key] = point
        if self.retain_set:
            self.retrain_retain_set()
        if self.compression_set:
            self.merge_compression_set()
        self.write_log()

    def wrap_up(self):
        for i in range(len(self.compression_set)):
            _, cluster = min([(self.MahalanobisDistance(self.compression_set[i].get_centroid(),\
                 self.discard_set[cluster].get_centroid(), self.discard_set[cluster].get_variance()),\
                      cluster) for cluster in range(len(self.discard_set))])
            while not self.compression_set[i].is_empty():
                key, point = self.compression_set[i].pop()
                self.discard_set[cluster].centroid_update(key, point)
        self.compression_set = []

        for rs in self.retain_set:
            _, cluster = min([(self.MahalanobisDistance(self.retain_set[rs],\
                 self.discard_set[cluster].get_centroid(), self.discard_set[cluster].get_variance()),\
                      cluster) for cluster in range(len(self.discard_set))])
            self.discard_set[cluster].centroid_update(rs, self.retain_set[rs])
        
        self.retain_set = {}
        self.log.pop()
        self.round -= 1
        self.write_log()

    def write_log(self):
        self.round += 1
        val1 = sum([self.discard_set[i].N for i in range(len(self.discard_set))])
        val2 = sum([self.compression_set[i].N for i in range(len(self.compression_set))])
        self.log.append("{},{},{},{},{},{}\n".format(self.round, len(self.discard_set), val1, len(self.compression_set), val2, len(self.retain_set)))
    
    def jsonify(self):
        res = {}
        for i in range(len(self.discard_set)):
            for key in self.discard_set[i].keys:
                res[str(key)] = i
        return json.dumps(res, sort_keys=True)
            

if __name__ == "__main__":

    MAGIC_NUMBER = 5

    def preprocess(path, idx_arr, matrix):
        count = 0
        with open(path, "r") as f:
            line  = f.readline()
            while line:
                arr = line.split(",")
                for i in range(len(arr)):
                    if i > 0:
                        arr[i] = float(arr[i])
                    else:
                        idx_arr.append(int(arr[i]))
                count += 1
                matrix.append(arr[1:])
                line = f.readline()
    
if __name__ == "__main__":
    
    input_dir = sys.argv[1]
    ncluster = int(sys.argv[2])
    output_path1 = sys.argv[3]
    output_path2 = sys.argv[4]

    filenum = 0
    bfr = None

    for dirname, _, files in os.walk(input_dir, topdown=False):
        for file in files:
            idx_arr, matrix = [], []
            preprocess(dirname + "/" + file, idx_arr, matrix)
            if filenum == 0:
                bfr = BFR(ncluster, MAGIC_NUMBER, idx_arr, matrix)
            else:
                bfr.update(idx_arr, matrix)
            filenum += 1
        bfr.wrap_up()
    
    with open(output_path1, "w") as f:
        f.write(bfr.jsonify())
    
    with open(output_path2, "w") as f:
        for log in bfr.log:
            f.write(log)
    

