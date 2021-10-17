import matplotlib.pyplot as plt
from scipy.io import arff
import pandas

data = arff.loadarff("/mnt/d/downloads/Apre/hw1/H01/scripts/data/breast.w.arff")
dataset = pandas.DataFrame(data[0])

#print(dataset)
#print(isinstance(dataset["Clump_Thickness"][0], float))

bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = [0.3, 0.2, 0.1, 0.5, 0.4, 0.6, 0.7, 0.5, 0.2, 0.1]

plt.hist(x, bins=bins)

plt.xlabel("Clump Thickness (value)")
plt.ylabel("probability")
plt.title("Clump Thickness")

plt.show()
