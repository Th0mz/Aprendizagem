from scipy.io import arff
import pandas
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

data = arff.loadarff("C:\\Users\\print\\Downloads\\kin8nm.arff")
dataset = pandas.DataFrame(data[0])

inputs = dataset.drop(columns=["y"]).values
outputs = dataset["y"].values

predictions1 = np.ndarray(shape = (0,))
predictions2 = np.ndarray(shape = (0,))
actual = np.ndarray(shape = (0,))

k_fold = KFold(n_splits=5, random_state= 0, shuffle=True)
classifier1 = MLPRegressor(solver = "sgd",alpha = 1, activation = "relu", hidden_layer_sizes=(3, 2), early_stopping = False,  random_state= 13, max_iter = 1500)
classifier2 = MLPRegressor(solver = "sgd",alpha = 1e-5, activation = "relu", hidden_layer_sizes=(3, 2), early_stopping = False,  random_state= 13, max_iter = 1500)

for train, test in k_fold.split(dataset):
    classifier1.fit(inputs[train], outputs[train])
    predicted1 = classifier1.predict(inputs[test])
    classifier2.fit(inputs[train], outputs[train])
    predicted2 = classifier2.predict(inputs[test])

    print(classifier1.score(inputs[test],outputs[test]))
    predictions1 = np.concatenate((predictions1,predicted1), axis= 0)
    predictions2 = np.concatenate((predictions2,predicted2), axis= 0)
    actual = np.concatenate((actual,outputs[test]), axis= 0)

residuals1 = actual - predictions1
residuals2 = actual - predictions2

residuals = [residuals1, residuals2]

fig, ax = plt.subplots()
ax.boxplot(residuals)

plt.show()

