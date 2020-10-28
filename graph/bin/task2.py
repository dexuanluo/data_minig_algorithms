import sys
import types
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import *
from itertools import combinations

def inverse_map(x):
    res = []
    for s in x[1]:
        res.append((s, (x[0], len(x[1]))))
    return res

THRESHOLD = 0.5
MAGIC_NUMBER = 10000000000000

class GraphSolver:
    """
    USAGE:
    vertices = [1,2,3]
    edges = [(1,2),(2,3),(1,3)]
    graph = GraphSolver(vertices, edges)                              
    graph.GirvanNewman(max_iter = 50, base = 1)
    res = graph.extract_communities()
    """
    def __init__(self, vertices, edges):
        self.vertices = vertices[:]
        self.edges = edges[:]
        self.vertices_map = {node : i for i, node in enumerate(vertices)}
        self.adjacency_matrix = [[0] * len(vertices) for _ in range(len(vertices))]
        self.betweenness_matrix = [[0] * len(vertices) for _ in range(len(vertices))]
        self.groups = [i for i in range(len(vertices))]
        self.degrees = [0] * len(vertices)
        self.best_modularity = -float("Inf")
        for x, y in edges:
            self.adjacency_matrix[self.vertices_map[x]][self.vertices_map[y]] = 1
            self._union(self.vertices_map[x], self.vertices_map[y])
            self.degrees[self.vertices_map[x]] += 1
            self.degrees[self.vertices_map[y]] += 1
        self._root_all_nodes()
        self.best_communities = self.groups[:]

    def __str__(self):
        res = "GraphSolver Adjacency Matrix\n"
        for row in self.adjacency_matrix:
            res += str(row) + "\n"
        return res

    def _groups_reinitialize(self):
        for i in range(len(self.groups)):
            self.groups[i] = i
        
    def _find(self, vertex):
        if vertex != self.groups[vertex]:
            self.groups[vertex] = self._find(self.groups[vertex])
        return self.groups[vertex]
    
    def _root_all_nodes(self):
        for i in range(len(self.groups)):
            self._find(i)
    
    def _union(self, vertex1, vertex2):
        p1 = self._find(vertex1)
        p2 = self._find(vertex2)
        if p1 != p2:
            self.groups[max(p1, p2)] = min(p1, p2)
            return True
        return False   
    def _get_bfs_levels(self, root):
        q = [root]
        seen = set(q)
        levels = []
        contribution = {root : 1}
        while q:
            levels.append(q)
            level = []
            level_d = set()
            for node in q:
                for child in range(len(self.adjacency_matrix[node])):
                    if self.adjacency_matrix[node][child] == 1:
                        if child not in seen and child not in level_d:
                            level.append(child)
                            seen.add(child)
                            level_d.add(child)
                            contribution[child] = contribution[node]
                        elif child in level_d:
                            contribution[child] += contribution[node]
            q = level
            
        return levels[::-1], contribution
    def _level_backward_traversal(self, levels, contribution):
        
        for i in range(len(levels) - 1):
            level = levels[i]
            next_level = set(levels[i + 1])
            for node in level:
                node_credits = sum(self.betweenness_matrix[node]) + 1
                for parent in next_level:
                    if self.adjacency_matrix[node][parent] == 1:                
                        self.betweenness_matrix[node][parent] += node_credits * (contribution[parent] / contribution[node])
                        self.betweenness_matrix[parent][node] += node_credits * (contribution[parent] / contribution[node])
            
        
    def get_betweenness(self, trueIndex = False):
        res = {}
        for vertex in self.vertices:
            levels, contribution = self._get_bfs_levels(self.vertices_map[vertex])
            self._level_backward_traversal(levels, contribution)
            for i in range(len(self.betweenness_matrix)):
                for j in range(i):
                    if self.betweenness_matrix[j][i] > 0:
                        if (j, i) not in res:
                            res[(j, i)] = 0
                        res[(j, i)] += self.betweenness_matrix[j][i]
            self._reset_betweenness()

        res_arr = []
        for key in res:
            a, b = key
            if trueIndex:
                res_arr.append([self.vertices[a], self.vertices[b], res[key] / 2])
            else:
                res_arr.append((a, b, res[key]))
        res_arr.sort(key = lambda x : (-x[2], x[0], x[1]))
        if trueIndex:
            return res_arr
        return res_arr
    
    def _reset_betweenness(self):
        for i in range(len(self.betweenness_matrix)):
            for j in range(len(self.betweenness_matrix[i])):
                self.betweenness_matrix[i][j] = 0

    def _cut_edge(self, vertex1, vertex2):
        self.adjacency_matrix[vertex1][vertex2] = 0
        self.adjacency_matrix[vertex2][vertex1] = 0
        self.degrees[vertex1] -= 1
        self.degrees[vertex2] -= 1

    def GirvanNewman(self ,max_iter = 1000, base = 2):
        count = 0
        while True:
            betweenness = self.get_betweenness()
            betweenness = betweenness[::-1]
            max_bet = betweenness[-1][2]

            while betweenness and abs(betweenness[-1][2] - max_bet) < 0.0000001:
                vertex1, vertex2, _ = betweenness.pop()
                self._cut_edge(vertex1, vertex2)
            
            self._groups_reinitialize()

            for vertex1, vertex2, _ in betweenness:
                self._union(vertex1, vertex2)

            self._root_all_nodes()

            modularity = self.get_modularity(betweenness, base)
            
            if modularity > self.best_modularity:
                self.best_communities = self.groups[:]
                self.best_modularity = modularity
            count += 1
            if not betweenness or count >= max_iter:
                break

    def get_modularity(self, betweenness, base):
        m =  len(betweenness)
        if m == 0: return -float("Inf")
        partition = {}
        modularity = {}
        for i in range(len(self.groups)):
            if self.groups[i] not in partition:
                partition[self.groups[i]] = []
                modularity[self.groups[i]] = 0
            partition[self.groups[i]].append(i)
        
        for group in partition:
            for pair in combinations(partition[group], 2):
                vertex1, vertex2 = pair
                modularity[group] += self.adjacency_matrix[vertex1][vertex2]\
                     - ((self.degrees[vertex1] * self.degrees[vertex2]) / (base * m))
        
        return sum([modularity[group] for group in modularity]) / (2 * m)
            
    def extract_communities(self):
        result_communities = {}
        for i in range(len(self.best_communities)):
            if self.best_communities[i] not in result_communities:
                result_communities[self.best_communities[i]] = []
            result_communities[self.best_communities[i]].append(self.vertices[i])

        res = [result_communities[com] for com in result_communities]
        res.sort(key = lambda x : len(x))

        return res




if __name__ == "__main__":
    
    argv = sys.argv
    CASE_NUM = int(argv[1])
    input_path = argv[2]
    betweenness_output_path = argv[3]
    community_output_path = argv[4]

    
    conf = SparkConf().setAppName("task2")\
        .setMaster("local[*]")\
            .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    

    sc = SparkContext(conf = conf)

    edges = sc.textFile(input_path)

    edges = edges\
        .map(lambda x : tuple(x.split(",")))\
            .filter(lambda x : x[0] != "user_id")

    all_users = edges\
        .map(lambda x : x[0])\
            .distinct()\
                .collect()

    all_users.sort()
    users_map = {user : i for i, user in enumerate(all_users)}

    all_states = edges\
        .map(lambda x : x[1])\
            .distinct()\
                .collect()

    all_states.sort()
    states_map = {states : i + MAGIC_NUMBER for i, states in enumerate(all_states)}

    edges = edges.map(lambda x : (users_map[x[0]], states_map[x[1]]))

    if CASE_NUM == 2:
        edges = edges\
            .groupByKey()\
                .map(lambda x : (x[0], tuple(x[1])))\
                    .flatMap(inverse_map)\
                        .groupByKey()\
                            .map(lambda x : (x[0], sorted(tuple(x[1]))))\
                                .flatMap(lambda x : [(pair, 1) for pair in combinations(x[1], 2)])\
                                    .reduceByKey(lambda x, y: x + y)\
                                        .map(lambda x : (x[0][0], x[0][1], x[1]))\
                                            .map(lambda x : (x[0][0], x[1][0], x[2] / (x[0][1] + x[1][1] - x[2])))\
                                                .filter(lambda x : x[2] >= THRESHOLD)\
                                                    .map(lambda x : x[:2])
    """
    Make DAG Doubly Linked
    """
    edges = edges.flatMap(lambda x : [x, x[::-1]])

    vertices = edges\
        .map(lambda x : x[0])\
            .distinct()

    edges_set = edges.collect()
    vertices_set = vertices.collect()     
    
    

    graph = GraphSolver(vertices_set, edges_set)

    if CASE_NUM == 2:
        betweenness = graph.get_betweenness(trueIndex = True)
        
        for i in range(len(betweenness)):
            a = min(all_users[betweenness[i][0]], all_users[betweenness[i][1]])
            b = max(all_users[betweenness[i][0]], all_users[betweenness[i][1]])
            betweenness[i][0] = a
            betweenness[i][1] = b
        betweenness.sort(key = lambda x : (-x[2], x[0], x[1]))

        with open(betweenness_output_path, "w") as f:
            for v1, v2, b in betweenness:
                f.write("('{}', '{}'), {}\n".format(v1, v2, b))
    
    
    res = []
    if CASE_NUM == 2:
        graph.GirvanNewman(max_iter = 50, base = 2)
        res = graph.extract_communities()

    else:
        graph.GirvanNewman(max_iter = 58, base = 1)
        res = graph.extract_communities()

    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = "'" + all_users[res[i][j]] + "'" if res[i][j] < MAGIC_NUMBER else "'" + all_states[res[i][j] - MAGIC_NUMBER] + "'"
        res[i].sort()

    res.sort(key = lambda x : (len(x), x))

    with open(community_output_path, "w") as f:
            for row in res:
                f.write(", ".join(row) + "\n")

