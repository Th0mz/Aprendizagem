from math import sqrt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from scipy.io import arff
from scipy.stats import ttest_rel
import pandas

accuracykNN3 = [0.9565217391304348, 
        0.927536231884058,
        0.9710144927536232,
        0.9558823529411765,
        0.9852941176470589,
        1.0,
        1.0,
        1.0,
        0.9558823529411765,
        0.9558823529411765
]


def mean(values):
    return sum(values) / len(values)

def standardDeviation (values):
    quadraticOffsets = []
    _mean = mean(values)
    for value in values:
        quadraticOffsets.append((value - _mean)**2)
    
    return sqrt(sum(quadraticOffsets) / len(values))

accuracyMeanKNN = mean(accuracykNN3)
accuracyStdDevKNN = standardDeviation(accuracykNN3)

print("kNN\n=============================")
print(f"mean = {accuracyMeanKNN}")
print(f"stdDev = {accuracyStdDevKNN}")
print(f"variancia = {accuracyStdDevKNN**2}")

data = arff.loadarff("C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\data\\breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values

k_fold = KFold(n_splits=10, random_state=13, shuffle=True)
accuracyNB = []

for train, test in k_fold.split(dataset):
    multinomialNBClassifier = MultinomialNB()
    multinomialNBClassifier = multinomialNBClassifier.fit(inputs[train], outputs[train])
    predicted = multinomialNBClassifier.predict(inputs[test])
    
    accuracy = 0
    hits = 0
    for i in range(len(predicted)):
        if (predicted[i] == outputs[test][i]):
            hits += 1

    accuracy = hits / len(predicted)
    accuracyNB.append(accuracy)

accuracyMeanNB = mean(accuracyNB)
accuracyStdDevNB = standardDeviation(accuracyNB)

print("\nNB\n=============================")
print(f"mean = {accuracyMeanNB}")
print(f"stdDev = {accuracyStdDevNB}")
print(f"variancia = {accuracyStdDevNB**2}")


print(len(accuracykNN3), len(accuracyNB))
ttest = ttest_rel(accuracykNN3, accuracyNB, alternative="greater")
pValue = ttest.pvalue
print(ttest, pValue)


