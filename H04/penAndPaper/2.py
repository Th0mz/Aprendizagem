from math import sqrt

def euclideanDistance (vector1, vector2):
    return sqrt((vector1[0] - vector2[0])**2 + (vector1[1] - vector2[1])**2)

dataset = [[[2, 4], [-1, 2], [4, 0]], [[-1, -4]]]
#dataset = [[[0, 0], [1, 0]], [[0, 2], [2, 2]]]

# a(X)
a = []
print("a(X)")
clusterIndex = 1
instanceIndex = 1
for cluster in dataset:
    print(f"===== cluster{clusterIndex} =====")
    for i in range(len(cluster)):
        aX = 0
        for j in range(len(cluster)):
            if (i != j):
                aX += euclideanDistance(cluster[i], cluster[j])
        if (len(cluster) == 1):
            aX = 1
        else:
            aX = aX / (len(cluster) - 1)
        a.append(aX)
        print(f"a(x{instanceIndex}) = {aX}")
        instanceIndex += 1
    clusterIndex += 1

# b(X)
b = []
print("\nb(X)")
instanceIndex = 1
for clusterIndex in range(len(dataset)):
    print(f"===== cluster{clusterIndex + 1} =====")
    for otherClusterIndex in range(len(dataset)):
        if (clusterIndex != otherClusterIndex):
            cluster = dataset[clusterIndex]
            otherCluster = dataset[otherClusterIndex]
            for instance in cluster:
                bX = 0
                for otherInstance in otherCluster:
                    bX += euclideanDistance(instance, otherInstance)
                bX = bX / len(otherCluster)
                b.append(bX)
                print(f"b(x{instanceIndex}) = {bX}")
                instanceIndex += 1

# S(X)
print("\nS(X)")
print("===================")
for i in range(len(a)):
    sX = 1 - a[i] / b[i]
    print(f"S(x{i + 1}) = {sX}")
