from   scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal as mNormal
import matplotlib.pyplot as plt
import numpy as np

dataset = [[ 2,  4],
           [-1, -4],
           [-1,  2],
           [ 4,  0]
]


c1Normal = mNormal(mean=[2, 4], cov=[[1, 0], [0, 1]])
c2Normal = mNormal(mean=[-1, -4], cov=[[2, 0], [0, 2]])

i = 1
print("=== Parte 1 ===")
print("determinar os valores das funções de densidade \nprobabilidade das multivariate para os pontos do dataset\n")
for value in dataset:
    print(f" === x{i} ===")
    print(f"P(x{i} | c1 = 1) = {c1Normal.pdf(value)}")
    print(f"P(x{i} | c2 = 1) = {c2Normal.pdf(value)}")

    i += 1


print("\n=== Parte 2 ===")
print("fazer sketch da forma do cluster")


mu1 = [1.565383248, 2.100727792]
sigma1 = [[4.132822984, -1.163367794], [-1.163367794, 2.605601057]]

mu2 = [-0.383703757, -3.41757815]
sigma2 = [[2.701660142, 2.106240599], [2.106240599, 2.169241946]]

x = []
y = []
for instance in dataset:
    x.append(instance[0])
    y.append(instance[1])

N    = 200
X    = np.linspace(-7, 7, N)
Y    = np.linspace(-7, 7, N)
X, Y = np.meshgrid(X, Y)
pos  = np.dstack((X, Y))

gaussian1 = multivariate_normal(mu1, sigma1)
gaussian2 = multivariate_normal(mu2, sigma2)
Z1    = gaussian1.pdf(pos)
Z2    = gaussian2.pdf(pos)

plt.scatter(x, y)
plt.contour(X, Y, Z1)
plt.contour(X, Y, Z2)
plt.show()
