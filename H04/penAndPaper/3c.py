import matplotlib.pyplot as plt

def vcMLP (N):
    return  (3 * (N**2)) + (5 * N) + 2

def vcBaysian (N):
    return (N**2) + (3 * N) + 1

m = [2, 5, 10, 30, 100, 300, 1000]

MLPpoints = []
baysianPoints = []
for dim in m:
    MLPpoints.append(vcMLP(dim))
    baysianPoints.append(vcBaysian(dim))

print(MLPpoints)  
print(baysianPoints) 

plt.plot(m, MLPpoints, color='y')
plt.plot(m, baysianPoints, color='g')

labels = ["MLP", "Baysian Classifier"]
plt.xlabel(f"m")
plt.ylabel("VC-dimension")
plt.legend(labels)
plt.show()