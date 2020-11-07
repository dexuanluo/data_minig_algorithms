import sys
import json
from random import sample, randint

class KMeans:
    def __init__(self, matrix, ncluster, random_init = True):
        self.matrix = matrix
        self.centroids = []
        self.ncluster = ncluster
        self.membership = None
        if random_init:
            self._random_init()
        else:
            self._centorids_init()
    
    def _random_init(self):
        self.membership = [randint(0, self.ncluster - 1) for _ in range(len(self.matrix))]
        self.centroids = [[0] * len(self.matrix[0]) for _ in range(self.ncluster)]
        self.renew_centroids()

    def _centorids_init(self, dis_func = None):
        if not dis_func:
            dis_func = self.EuclideanDistance
        self.membership = [-1 for _ in range(len(self.matrix))]
        pool = set([i for i in range(len(self.membership))])
        count = 1
        last_idx = sample(pool, 1)[0]
        last = self.matrix[last_idx]
        self.centroids.append(last)
        pool.remove(last_idx)

        while count < self.ncluster:
            max_dis = -float("Inf")
            max_idx = -1
            candidate = None
            for i, point in enumerate(self.matrix):
                dis = dis_func(point, last)
                if dis > max_dis:
                    max_dis = dis
                    max_idx = i
                    candidate = point
            if max_idx in pool:
                self.centroids.append(candidate)
                last_idx = max_idx
                last = candidate
                pool.remove(last_idx)
            else:
                last_idx = sample(pool, 1)[0]
                self.centroids.append(self.matrix[last_idx])
                last = self.matrix[last_idx]
                pool.remove(last_idx)
            count += 1
        

    def kmeans_train(self, max_iter = 1000, tolerance = 0, dis_func = None):
        if not dis_func:
            dis_func = self.EuclideanDistance
        for _ in range(max_iter):
            res = self.renew_membership(dis_func, tolerance)
            self.renew_centroids()
            if res: break
        
    def _reset_centroids(self):
        for i in range(len(self.centroids)):
            for j in range(len(self.centroids[i])):
                self.centroids[i][j] = 0.0
    
    def renew_centroids(self):
        self._reset_centroids()
        counter = {i : 0 for i in range(self.ncluster)}
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                self.centroids[self.membership[i]][j] += self.matrix[i][j]
            counter[self.membership[i]] += 1
        
        for i in range(len(self.centroids)):
            for j in range(len(self.centroids[i])):
                self.centroids[i][j] /= (counter[i] + 0.00000001)
        
        
    def renew_membership(self, dis_func, tolerance):
        diff = 0
        for i in range(len(self.matrix)):
            _, cluster = min([(dis_func(self.matrix[i], centroid), j) for j, centroid in enumerate(self.centroids)])
            diff += 1 if cluster != self.membership[i] else 0
            self.membership[i] = cluster
        
        return diff <= tolerance

    def EuclideanDistance(self, point1, point2):
        return sum([(x - y)**2 for x, y in zip(point1, point2)])**0.5

    def jsonify(self):
        s = {str(i):self.membership[i] for i in range(len(self.membership))}
        return json.dumps(s)


if __name__ == "__main__":
    ncluster = int(sys.argv[2])
    input_path = sys.argv[1]
    output_path = sys.argv[3]

    matrix = []
    
    with open(input_path, "r") as f:
        line  = f.readline()
        while line:
            arr = line.split(",")
            
            for i in range(len(arr)):
                arr[i] = float(arr[i])
            
            matrix.append(arr[1:])
            
            line = f.readline()
    
    
    kmeans = KMeans(matrix, ncluster, random_init=False)

    kmeans.kmeans_train(max_iter=1000)
    
    with open(output_path, "w") as f:
        f.write(kmeans.jsonify())