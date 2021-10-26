from scipy.io import arff
import pandas
from sklearn import tree
from sklearn.model_selection import KFold
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
    for train, test in k_fold.split(dataset):
        treeClassifier = tree.DecisionTreeClassifier(criterion="entropy", max_features=maxFeatures)
        treeClassifier.fit(inputs[train], outputs[train])

        trainingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputs[train], outputs[train]))
        testingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputs[test], outputs[test]))

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















I. Pen-and-paper

	First, we transformed the design matrix using the basis function
 X=\left[\begin{matrix}1&1&1&0\\1&1&1&5\\1&0&2&4\\1&1&2&3\\1&2&0&7\\1&1&1&1\\1&2&0&2\\1&0&2&9\\\end{matrix}\right]{{\rightarrow\below{\phi\left(x\right)}}\ \ \Phi=\left[\begin{matrix}1&1.414&2&2.828\\1&5.196&27&140.3\\1&4.472&20&89.44\\1&3.742&14&52.38\\1&7.280&53&385.8\\1&1.732&3&5.196\\1&2.828&8&22.63\\1&9.220&85&783.7\\\end{matrix}\right]}      \Phi^T=\ \left[\begin{matrix}1&1&1&1&1&1&1&1\\1.41&5.196&4.47&3.74&7.28&1.7&2.8&9.22\\2&27&20&14&53&3&8&85\\2.82&140.3&89.4&52.4&386&5.2&23&784\\\end{matrix}\right]\    
Then, we will calculate the vector W that minimizes the square-error loss function, this vector is given by the following expression
W=\left(\Phi^T\Phi\right)^{-1}\Phi^T\mathrm{\ \ z\ where\ z\ \ }=[13206457] Toutput vector
So, for the training dataset we obtain that W=[4.5835-1.68720.3377-0.01331]T
	To measure the differences between the observed values and the predictions given by the model, in order to test it, we’ll use the RMSE
For that we’ll need to calculate the predictions of our model:
X_{train}=\left[\begin{matrix}1&2&0&0\\1&1&2&1\\\end{matrix}\right]{{\rightarrow\below{\phi\left(x\right)}}\ \Phi}_{train}=\left[\begin{matrix}1&2&4&8\\1&2.449&6&14.6969\\\end{matrix}\right]
Each prediction is given by the following polynomial regression model \hat{z}=\Phi^t.w=\left[\begin{matrix}2.4536&2.2816\\\end{matrix}\right]^T
Therefore, the RMSE is:
RMSE=\sqrt{\frac{\sum_{i=9}^{10}\left(z_i-\widehat{z_i}\right)^2}{2}}=\sqrt{\frac{\left(2-2.4536\right)^2+\left(4-2.2816\right)^2}{2}}=1.256

	y_{3_{new}}	t
x_1	0	N
x_2	1	N
x_3	1	N
x_4	0	N
x_5	1	P
x_6	0	P
x_7	0	P
x_8	1	P
P\left(y_1=0\right)	0.25
P\left(y_1=1\right)	0.5
P\left(y_1=2\right)	0.25
P\left(t=N\middle| y_1=0\right)	0.5
P\left(t=N\middle| y_1=1\right)	0.75
P\left(t=N\middle| y_1=2\right)	0
P\left(t=P\middle| y_1=0\right)	0.5
P\left(t=P\middle| y_1=1\right)	0.25
P\left(t=P\middle| y_1=2\right)	1
P\left(y_2=0\right)	0.25
P\left(y_2=1\right)	0.375
P\left(y_2=2\right)	0.375
P\left(t=N\middle| y_2=0\right)	1
P\left(t=N\middle| y_2=1\right)	0.666667
P\left(t=N\middle| y_2=2\right)	0.666667
P\left(t=P\middle| y_2=0\right)	0
P\left(t=P\middle| y_2=1\right)	0.333333
P\left(t=P\middle| y_2=2\right)	0.333333
P\left(t=N\middle| y_3=0\right)	0.5
P\left(t=N\middle| y_3=1\right)	0.5
P\left(t=N\middle| y_3=0\right)	0.5
P\left(t=N\middle| y_3=1\right)	0.5
P\left(t=P\middle| y_3=0\right)	0.5
P\left(t=P\middle| y_3=1\right)	0.5
	Firstly, we computed the equal depth binarization of y_3 , in which we used the median of its train values as the criteria for the binarization, y_3\ median=3.5, and the class targets t_i, so we obtained the following table:
 To learn a decision tree using ID3, we need to calculate the information gain (IG) of each variable, which is given by IG=H\left(t\right)-H\left(t\middle| y_i\right)\ where\ i=\left\{1,2,3\right\}, and H the entropy.
Probabilities of the training dataset:






-H\left(t\middle| y_1\right)=H\left(t\middle| y_1=0\right)+H\left(t\middle| y_1=1\right)+H\left(t\middle| y_1=2\right)=\ 0.25\ +\ 0.405639\ +\ 0\ =\ 0.655639\ \ 
- H\left(t\middle| y_2\right)=H\left(t\middle| y_2=0\right)+H\left(t\middle| y_2=1\right)+H\left(t\middle| y_2=2\right)=\ 0\ +\ 0.344361\ +\ 0.344361\ \ =\ 0.688722\ \ 
- H\left(t\middle| y_3\right)=H\left(t\middle| y_3=0\right)+H\left(t\middle| y_3=1\right)\ =\ 0.5\ +\ 0.5\ =\ 1  

Information Gain: IG\left(y_1\right)=1-0.655639=0.344361;IG\left(y_1\right)=1-0.688722=0.311278;\bigm-IG\left(y_3\right)=1-1=0


We can conclude that the variable that will be used as root for the decision tree is y_1 as it is the variable with the highest IG, and y_1=2 always leads to t\ =\ P, and y_0=0 to uncertain (?). So, we will need to study now only the cases that y_1=1, which are on the following table:
 	y2	y3	t
x1	1	0	N
x2	1	1	N
x4	2	0	N
x6	1	0	P

From this, we can conclude that y_2=0 is uncertain, and that y_2=2 leads to t\ =\ N. Since the information gain of y_2 and y_3 is the same (using analogous calculus from above), we choose the add y_2 to the decision tree, which leaves us with the final table:
	y3	t
x1	0	0
x2	1	0
x6	0	1
 
Here, we can conclude that when y_3=1 , then t\ =\ 0, and with analogous methods, it is clear that when y_3=0 the information gain will be 0, so that will lead to uncertainty. With all this information, we are now able to create the decision tree:

	Using the “test” dataset on the decision tree obtained earlier (x_9 and x_{10}) we get a prediction t_9=P, t_{10}=N(by following the branch), opposed to the true values of t_9=N, t_{10}=P, so, the accuracy will be: 
accuracy=\frac{#\mathrm{true\ positives}+#\mathrm{true\ negatives}}{#\mathrm{total\ observations}}=\frac{0}{2}=0%



II. Programming and critical analysis

	Using 10-fold cross validation for splitting the data into training and testing data, we trained our decision tree model in two different ways: 
      i) \mathrm{number\ of\ selected\ features}\in\{1,3,5,9} 
      ii) maximum\ tree\ depth\in\{1,3,5,9}
After fitting the model, the score is calculated for both training and testing data. The score of each model can be compared in the given plots:
 

	In both graphs we can observe that the values of the training and testing accuracy are somewhat correlated, 
	Pila
	Pila

	Podemos observar que a medida que aumentamos a profundidade, o risco de overfitting aumenta, já que o modelo começa a aprender demasiado bem o training set (accuracy da training data vai aumentando), até certo ponto a accuracy do test data acompanha o aumento da training data, mas a certo (maximum tree depth = 5) desce ligeiramente o que nos leva a concluir que o modelo não está a aprender realmente o problema mas as nuances da training data (overfitting). O que nos leva a concluir que a melhor escolha para a produndidade será de 5 (best fit)













dataset = pandas.DataFrame(arff.loadarff(<path>)[0])
dataset["Class"] = dataset["Class"].str.decode('utf-8')
inputs = dataset.drop(columns=["Class"]).values; outputs = dataset["Class"].values
k_fold = KFold(n_splits=10, random_state=13, shuffle=True)

def calculateDictMean(dictionary):
	return [sum(dictionary[key])/len(dictionary[key]) for key in dictionary]

trainingAccuracy_i, testingAccuracy = {"1" : [], "3" : [], "5": [], "9": []}
for maxFeatures in [1, 3, 5, 9]:
    for train, test in k_fold.split(dataset):
        treeClassifier = tree.DecisionTreeClassifier(criterion="entropy", max_features=maxFeatures)
        treeClassifier.fit(inputs[train], outputs[train])

        trainingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputs[train], outputs[train]))
        testingAccuracy_i[str(maxFeatures)].append(treeClassifier.score(inputs[test], outputs[test]))

trainingAccuracyMean_i, testingAccuracyMean_i = calculateDictMean(trainingAccuracy_i), calculateDictMean(testingAccuracy_i)

trainingAccuracy_ii, testingAccuracy_ii = {"1" : [], "3" : [], "5": [], "9": []}
for depth in [1, 3, 5, 9]:
    for train, test in k_fold.split(dataset):
        treeClassifier = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        treeClassifier.fit(inputs[train], outputs[train])

        trainingAccuracy_ii[str(depth)].append(treeClassifier.score(inputs[train], outputs[train]))
        testingAccuracy_ii[str(depth)].append(treeClassifier.score(inputs[test], outputs[test]))

trainingAccuracyMean_ii, testingAccuracyMean_ii = calculateDictMean(trainingAccuracy_ii), calculateDictMean(testingAccuracy_ii)

plt.plot([1, 3, 5, 9], trainingAccuracyMean_i); plt.plot([1, 3, 5, 9], testingAccuracyMean_i)
plt.show(); plt.clf()

plt.plot([1, 3, 5, 9], trainingAccuracyMean_ii); plt.plot([1, 3, 5, 9], testingAccuracyMean_ii)
plt.show()




