from scipy.io import arff
import pandas
import numpy as np
import seaborn as sns
from sklearn import feature_selection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest
import seaborn as sns
import matplotlib.pyplot as plt

def ECR(labels, outputs, k):
    benign = [0, 0, 0]
    malign = [0, 0, 0]
    aux = 0

    for cluster in range(0, k):
        for i in range(0, len(labels)):
            if labels[i] == cluster:
                if outputs[i] == "benign":
                    benign[cluster]+=1
                else:
                    malign[cluster]+=1
    
    for cluster in range(0, k):
        max_val = max(benign[cluster], malign[cluster])

        aux += benign[cluster] + malign[cluster] - max_val

    return aux/k


data = arff.loadarff("C:\\Users\\print\\Downloads\\breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values


kMeans2 = KMeans(n_clusters = 2, random_state = 13)
kMeans3 = KMeans(n_clusters = 3, random_state = 13)

kMeans2.fit(inputs)
kMeans3.fit(inputs)

label2 = kMeans2.predict(inputs)
label3 = kMeans3.predict(inputs)

print(ECR(label2, outputs, 2))
print(ECR(label3, outputs, 3))

print(f"Silhouette score n=2: {silhouette_score(inputs, label2)}")
print(f"Silhouette score n=3: {silhouette_score(inputs, label3)}")

#clusters are not clearly apart from each other since is in the range of 0.5 and 0.6
#error classification rate

#5 usar selectkbest e fazer plot

#ECR fazer com o que esta nos slides das bolinhas e usar outputs

kbest = SelectKBest(feature_selection.mutual_info_classif, k = 2).fit(inputs, outputs)

inputsTopFeatures = inputs[:, kbest.get_support(indices = True)]

newKMeans3 = KMeans(n_clusters = 3, random_state = 13).fit(inputsTopFeatures)

plt.scatter(inputsTopFeatures[:,0], inputsTopFeatures[:,1], c = newKMeans3.labels_, cmap = 'plasma')
plt.show()
