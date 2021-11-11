from scipy.stats import norm as normal
from scipy.stats import multivariate_normal as mNormal
from math import sqrt

dataset = [[ 2,  4],
           [-1, -4],
           [-1,  2],
           [ 4,  0]
]


c1Normal = mNormal(mean=[2, 4], cov=[[1, 0], [0, 1]])
c2Normal = mNormal(mean=[-1, -4], cov=[[2, 0], [0, 2]])

i = 1
for value in dataset:
    print(f" === x{i} ===")
    print(f"P(x{i} | c1 = 1) = {c1Normal.pdf(value)}")
    print(f"P(x{i} | c2 = 1) = {c2Normal.pdf(value)}")

    i += 1