from scipy.io import arff
import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

#MUDAR PATH PARA OUTRO PC
data = arff.loadarff("/mnt/d/downloads/Apre/hws/Aprendizagem/H01/scripts/data/breast.w.arff")
dataset = pandas.DataFrame(data[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')

inputs = dataset.drop(columns=["Class"]).values
outputs = dataset["Class"].values

predictions = []
actual = []

k_fold = KFold(n_splits=5, random_state=0, shuffle=True)
classifier = MLPClassifier(activation = "relu", alpha=1e-5, hidden_layer_sizes=(3, 2), early_stopping = True)

for train, test in k_fold.split(dataset):
    classifier.fit(inputs[train], outputs[train])
    predicted = classifier.predict(inputs[test])

    predictions.append(predicted)
    actual.append(outputs[test])

cm = confusion_matrix(actual,predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)

disp.plot()

plot.show()
