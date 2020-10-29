import sys
import json
from random import sample, randint

class KMeans:
    def __init__(self, matrix, ncluster):
        self.matrix = matrix
        self.centroids = [[0] * len(matrix[0]) for _ in range(ncluster)]
        self.ncluster = ncluster
        self.membership = [randint(0, ncluster - 1) for _ in range(len(matrix))]

    def train(self, max_iter = 50, tolerance = 0, dis_func = None):
        if not dis_func:
            dis_func = self.EuclideanDistance
        for _ in range(max_iter):
            self.renew_centroids()
            res = self.renew_membership(dis_func, tolerance)
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
                self.centroids[i][j] /= (counter[i] + 0.00000000001)
        
        
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
            matrix.append(arr)
            line = f.readline()

    kmeans = KMeans(matrix, ncluster)

    kmeans.train(max_iter=1000)

    with open(output_path, "w") as f:
        f.write(kmeans.jsonify())