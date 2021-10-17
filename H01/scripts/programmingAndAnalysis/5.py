import matplotlib
from scipy.io import arff
import pandas

data = arff.loadarff("C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\Uni\\3Ano\\1Semestre\\Aprendizagem\\homeworks\\H01\\scripts\\data\\breast.w.arff")
dataset = pandas.DataFrame(data[0])

print(dataset)
#usar função hist