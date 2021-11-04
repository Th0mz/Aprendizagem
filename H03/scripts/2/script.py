from scipy.io import arff
import pandas
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

data = arff.loadarff("C:\\Users\\print\\Downloads\\breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values

predictions1 = np.ndarray(shape = (0,))
predictions2 = np.ndarray(shape = (0,))

actual = np.ndarray(shape = (0,))

k_fold = KFold(n_splits=5, random_state= 0, shuffle=True)
classifier1 = MLPClassifier(solver = "sgd",alpha = 1, activation = "relu", hidden_layer_sizes=(3, 2), early_stopping = False,  random_state= 13, max_iter = 1500)
classifier2 = MLPClassifier(solver = "sgd",alpha = 1, activation = "relu", hidden_layer_sizes=(3, 2), early_stopping = True,  random_state= 13, max_iter = 1500)

for train, test in k_fold.split(dataset):
    classifier1.fit(inputs[train], outputs[train])
    classifier2.fit(inputs[train], outputs[train])
    
    predicted1 = classifier1.predict(inputs[test])
    predicted2 = classifier2.predict(inputs[test])

    print(classifier1.score(inputs[test],outputs[test]))

    predictions1 = np.concatenate((predictions1,predicted1), axis= 0)
    predictions2 = np.concatenate((predictions2,predicted2), axis= 0)
    actual = np.concatenate((actual,outputs[test]), axis= 0)


print("=============CONFUSION==============")
cm1 = confusion_matrix(actual, predictions1, labels = classifier1.classes_)
display1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels = classifier1.classes_)

display1.plot()
display1.ax_.set(title = "Absence of early stopping", xlabel = "True", ylabel = "Predicted")

cm2 = confusion_matrix(actual, predictions2, labels = classifier2.classes_)
display2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels = classifier2.classes_)

display2.plot()
display2.ax_.set(title = "Presence of early stopping", xlabel = "True", ylabel = "Predicted")
plt.show()