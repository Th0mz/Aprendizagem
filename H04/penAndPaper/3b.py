import matplotlib.pyplot as plt

def vcMLP (N):
    return  (3 * (N**2)) + (5 * N) + 2

def vcTree (N):
    return 3**N

def vcBaysian (N):
    return (N**2) + (3 * N) + 1

m = [2, 5, 10, 12, 13]

MLPpoints = []
treePoints = []
baysianPoints = []
for dim in m:
    baysianPoints.append(vcBaysian(dim)) 
    MLPpoints.append(vcMLP(dim))
    treePoints.append(vcTree(dim))

plt.plot(m, MLPpoints, color='c')
plt.plot(m, baysianPoints, linestyle='--', color='g')
plt.plot(m, treePoints, color='y')

labels = ["MLP", "Baysian Classifier", "Decision Tree"]
plt.xlabel(f"m")
plt.ylabel("VC-dimension")
plt.legend(labels)
plt.show()