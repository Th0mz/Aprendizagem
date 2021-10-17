import matplotlib.pyplot as plt
from scipy.io import arff
import pandas

data = arff.loadarff("C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\data\\breast.w.arff")
dataset = pandas.DataFrame(data[0])

variables = ["Clump_Thickness", "Cell_Size_Uniformity", "Cell_Shape_Uniformity", "Marginal_Adhesion", "Single_Epi_Cell_Size", "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]
lines = len(dataset["Bare_Nuclei"])



for variable in variables:
    benign = []
    malignant = []
    name = variable.replace("_", " ")

    for i in range(lines):
        if (isinstance(dataset[variable][i], float) and 1.0 <= dataset[variable][i] <= 10.0):
            if (dataset["Class"][i] == b'malignant'):
                malignant.append(dataset[variable][i])

            elif (dataset["Class"][i] == b'benign'):
                benign.append(dataset[variable][i])

    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    bottom = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    labels = ["benign", "malignant"]
    plt.hist([benign, malignant] , bins=bins, density=True, alpha=0.5, align="left")


    plt.xlabel(f"{name} (value)")
    plt.ylabel("Probability")
    plt.legend(labels)
    plt.xticks(bottom)
    plt.title(f"{name} conditional to Class")

    plt.savefig(f"C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\programmingAndAnalysis\\5\\results5\\{name}.png")
    plt.clf()

#plt.show()
