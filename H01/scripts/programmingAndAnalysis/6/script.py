from sklearn.model_selection import KFold
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
import pandas


data = arff.loadarff("C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\data\\breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values



k_fold = KFold(n_splits=10, random_state=13, shuffle=True)
accuracyObserved = {"3": [], 
                    "5": [], 
                    "7": []}

for numNeighbors in range(3, 8, 2):
    for train, test in k_fold.split(dataset):
        knnClassifier = KNeighborsClassifier(n_neighbors = numNeighbors, weights="uniform")
        knnClassifier = knnClassifier.fit(inputs[train], outputs[train])
        predicted = knnClassifier.predict(inputs[test])

        accuracy = 0
        hits = 0
        for i in range(len(predicted)):
            if (predicted[i] == outputs[test][i]):
                hits += 1

        accuracy = hits / len(predicted)
        accuracyObserved[str(numNeighbors)].append(accuracy)

for numNeighbors in accuracyObserved:
    averageAccuracy = sum(accuracyObserved[numNeighbors]) / len(accuracyObserved[numNeighbors])
    print(f"average accuracy k={numNeighbors} => {averageAccuracy}")

print(accuracyObserved["3"])
