import random


class KMeansClustering:
    def __init__(self, k=5, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = {}
        print(f"K Means Clustering Parameters:\n"
              f"k: {self.k}\n"
              f"max_iters: {max_iters}")

    def train(self, data):
        data = [x.split(' ') for x in data]
        self.centroids = [data[random.randint(0, len(data)-10)] for i in range(self.k)]
        self.clusters = {i: [self.centroids[i], set(), 0] for i in range(self.k)}
        convergence_threshold = 0
        for i in range(1, self.max_iters + 1):
            print(f"Iteration: {i}", end="", flush=True)
            clusters = {i: [self.centroids[i], set(), 0] for i in range(self.k)}
            prev_centroids = self.centroids
            break_ = False
            # Assign Clusters
            for point in data:
                cluster_id, dist = self.predict(point)
                clusters[cluster_id][1].add(frozenset(point))
                clusters[cluster_id][2] += dist

            # Update Centroids
            for cluster_id in clusters:
                centroid = clusters[cluster_id][0]
                min_dist = clusters[cluster_id][2]
                for point1 in clusters[cluster_id][1]:
                    dist = 0
                    for point2 in clusters[cluster_id][1]:
                        dist += self.jaccard_distance(point1, point2)
                    if min_dist > dist:
                        min_dist = dist
                        centroid = point1

                clusters[cluster_id][0] = list(centroid)
                self.centroids[cluster_id] = list(centroid)
            if self.centroids == prev_centroids:
                if convergence_threshold > 3:
                    break_ = True
                else:
                    convergence_threshold += 1
            self.clusters = clusters
            print(f"\rIteration: {i} Error: {self.__get_sse()}")
            if break_: break
        return self.__get_sse()

    def predict(self, data):
        min_dist = float("inf")
        cluster_id = -1
        for cluster_index in self.clusters:
            cur_dist = self.jaccard_distance(data, self.centroids[cluster_index])
            if min_dist > cur_dist:
                cluster_id = cluster_index
                min_dist = cur_dist
        return cluster_id, min_dist

    def __get_sse(self):
        error = 0
        for cluster_id in self.clusters:
            for point in self.clusters[cluster_id][1]:
                error += self.jaccard_distance(point, self.clusters[cluster_id][0]) ** 2
        return error

    @staticmethod
    def jaccard_distance(t1, t2):
        intersection = set(t1).intersection(t2)
        union = set().union(t1, t2)
        return 1 - (len(intersection) / len(union))
