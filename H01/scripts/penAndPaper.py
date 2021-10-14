from scipy.stats import norm as normal
from scipy.stats import multivariate_normal as mNormal
from math import sqrt

def createInstance (y1, y2, y3, y4, _class):
    """dataset instance generator"""

    return {"y1" : y1, 
            "y2" : y2,
            "y3" : y3,
            "y4" : y4,
            "class" : _class}

# dataset representation
dataset = [
    createInstance( 0.6, "A",  0.2,  0.4, 0),
    createInstance( 0.1, "B", -0.1, -0.4, 0),
    createInstance( 0.2, "A", -0.1,  0.2, 0),
    createInstance( 0.1, "C",  0.8,  0.8, 0),
    createInstance( 0.3, "B",  0.1,  0.3, 1),
    createInstance(-0.1, "C",  0.2, -0.2, 1),
    createInstance(-0.3, "C", -0.1,  0.2, 1),
    createInstance( 0.2, "B",  0.5,  0.6, 1),
    createInstance( 0.4, "A", -0.4, -0.7, 1),
    createInstance(-0.2, "C",  0.4,  0.3, 1)
]

y3y4_c0 = mNormal(mean=[0.2, 0.25], cov=[[0.18, 0.18], [0.18, 0.25]])
y3y4_c1 = mNormal(mean=[0.1166667, 0.083333], cov=[[0.1096667, 0.1221333], [0.1221333, 0.2136667]])

i = 1
for instance in dataset:
    y3 = instance["y3"]
    y4 = instance["y4"]
    print(f"x_{i} :  P(y3 = {y3}, y4 = {y4} | class = 0) = {y3y4_c0.pdf([y3, y4])}")

    i += 1

"""
P(class=0| x1)= P(x1|class=0)P(class=0) / P(x1) = P(x1|class=0)P(class=0) =  P(y1=0.6|class=0) * P(y2="A"|class=0) * P(y3y4=[0.2, 0.4]|class=0) * P(class=0) / P(x1)

P(x1|class=0) = P(y1=0.6|class=0) * P(y2="A"|class=0) * P(y3y4=[0.2, 0.4]|class=0)
P(x1|class=1) = P(y1=0.6|class=1) * P(y2="A"|class=1) * P(y3y4=[0.2, 0.4]|class=1)
"""