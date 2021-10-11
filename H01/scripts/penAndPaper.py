from scipy.stats import norm as normal
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

def getVariableData (variable):
    """get all dataset data from the variable passed in the argument"""

    # check if the variable is part of dataset
    if (variable not in dataset[0].keys()):
        return None

    data = []
    for instace in dataset:
        data = data + [instace[variable]]
    
    return data

def getNormal(variable):
    """assuming that the values under the variable are distributed 
           in a normal way, get the normal parameters"""

    data = getVariableData(variable)

    # calculate data mean
    mean = round(sum(data) / len(data), 3)

    # calculate data standard deviation
    qdrDev = []
    for value in data:
        # calculate the quadratic deviation (x - mean)^2
        qdrDev = qdrDev + [(value - mean)**2]
    stdDev = round(sqrt(sum(qdrDev) / len(qdrDev)), 3)

    #Debug : print(mean, stdDev)
    return normal(mean, stdDev)


            
print(getNormal("y1").cdf(0.13))
print(getNormal("y1").pdf(0.13))
