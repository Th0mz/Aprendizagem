from scipy.io import arff
import pandas
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt


data = arff.loadarff("C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\data\\breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values

k_fold = KFold(n_splits=10, random_state=13, shuffle=True)

def calculateDictMean(dictionary):
    return [sum(dictionary[key])/len(dictionary[key]) for key in dictionary]


# i. number of selected features in {1,3,5,9} using mutual information (tree with no fixed depth)

trainingAccuracy_i = {"1" : [], "3" : [], "5": [], "9": []}
testingAccuracy_i = {"1" : [], "3" : [], "5": [], "9": []}
for maxFeatures in [1, 3, 5, 9]:
    inputsNew = SelectKBest(mutual_info_classif, k=maxFeatures).fit_transform(inputs ,outputs)
    for train, test in k_fold.split(dataset):
        treeClassifier = tree.DecisionTreeClassifier(criterion="entropy", max_features=maxFeatures)
        treeClassifier.fit(inputsNew[train], outputs[train])

        trainingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputsNew[train], outputs[train]))
        testingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputsNew[test], outputs[test]))

trainingAccuracyMean_i, testingAccuracyMean_i = calculateDictMean(trainingAccuracy_i), calculateDictMean(testingAccuracy_i)

print("5.")
print("i)")
print(trainingAccuracyMean_i)
print(testingAccuracyMean_i)

# ii. maximum tree depth in {1,3,5,9} (with all features and default parameters)

trainingAccuracy_ii, testingAccuracy_ii = {"1" : [], "3" : [], "5": [], "9": []}, {"1" : [], "3" : [], "5": [], "9": []}
for depth in [1, 3, 5, 9]:
    for train, test in k_fold.split(dataset):
        treeClassifier = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        treeClassifier.fit(inputs[train], outputs[train])

        trainingAccuracy_ii[str(depth)].append(treeClassifier.score(inputs[train], outputs[train]))
        testingAccuracy_ii[str(depth)].append(treeClassifier.score(inputs[test], outputs[test]))


trainingAccuracyMean_ii, testingAccuracyMean_ii = calculateDictMean(trainingAccuracy_ii), calculateDictMean(testingAccuracy_ii)

print("ii)")
print(trainingAccuracyMean_ii)
print(testingAccuracyMean_ii)


plt.plot([1, 3, 5, 9], trainingAccuracyMean_i)
plt.plot([1, 3, 5, 9], testingAccuracyMean_i)

plt.xlabel(f"Number of Selected Features")
plt.ylabel("Accuracy")

labels = ["train", "test"]
plt.legend(labels)
plt.title("i)")

plt.show()
plt.clf()

plt.plot([1, 3, 5, 9], trainingAccuracyMean_ii)
plt.plot([1, 3, 5, 9], testingAccuracyMean_ii)

plt.xlabel(f"Maximum Tree Depth")
plt.ylabel("Accuracy")

labels = ["train", "test"]
plt.legend(labels)
plt.title("ii)")

plt.show()