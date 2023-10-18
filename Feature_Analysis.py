from pyexcel_ods3 import save_data
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def load_file(path):
    xls = pd.read_csv(path)
    #df1 = pd.read_excel(xls, 'file')
    return xls

def seperate_data(feature):
    Symptomatic = []
    Asymptomatic = []
    hej = file[feature].values
    hej2 = file["S/AS"].values
    for i in range(len(hej2)):
        if hej2[i] == "S":
            Symptomatic.append(hej[i])
        elif hej2[i] == "AS":
            Asymptomatic.append(hej[i])
        else:
            print("Wrong in seperate_data function")
    Symptomatic = np.array(Symptomatic)
    Asymptomatic = np.array(Asymptomatic)
    return Symptomatic,Asymptomatic      

file = load_file(r"E:\file.csv")
#file = load_file(r"C:\Users\Simon\Downloads\BiKE_imported_csv.xlsx")
list_of_features = [
"S/AS", #0
"Total Calcification Volume", #1
"Number of calcifications", #2
"Largerst Calcification Volume", #3
"Mean Calcification volume", #4
"Maximum Calcification Arc", #5
"Mean Calcification Arc", #6
"Total Calcification Surface Area", #7
"Total Calcification Surface/Volume Ratio", #8
"Total Calcification Elongation", #9
"Total Calcification Flatness", #10
"Total Calcification Sphereicity", #11
"Largest Calcification Surface Area", #12
"Largest Calcification Surface/Volume Ratio", #13
"Largest Calcification Elongation", #14
"Largest Calcification Flatness", #15
"Largest Calcification Sphereicity", #16
"Mean Calcification-Lumen Distance", #17
]

def Scatter_plot(feature1,feature2):
    S,AS = seperate_data(list_of_features[feature1])
    S2,AS2 = seperate_data(list_of_features[feature2])
    plt.scatter( S, [0] * S.shape[0], c="Red")
    plt.scatter(AS, [1] * AS.shape[0], c="Blue")
    plt.scatter( S, S2, c="Red")
    plt.scatter(AS, AS2, c="Blue")
    plt.show()

def regression(feature1,feature2,classes):
    sb.lmplot(data=file, x=feature1, y=feature2, hue=classes)
    stats = stats.pearsonr(file['Number of calcifications'], file['Total Calcification Volume'])
    print(stats)
    plt.show()

def heatmap(file):
    
    data = file.iloc[:,2:]
    cormat = data.corr()
    round(cormat,2)
    sb.heatmap(cormat)
    plt.show()

def PCA_test(filename):
    
    pca = PCA(n_components=2)
    data = filename.iloc[:,2:]
    data = data.dropna()
    X_r = pca.fit(data).transform(data)
    y = data["S/AS"].values
    print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
    )
    plt.figure()
    colors = ["navy", "turquoise"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], "S/AS"):
        plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.show()

def LDA_test(filename):

    lda = LinearDiscriminantAnalysis(n_components=2)
    data = filename.iloc[:,2:]
    data = data.dropna()
    y = data["S/AS"].values
    X_r2 = lda.fit(data, y).transform(data)
#PCA_test(file)
#heatmap(file)