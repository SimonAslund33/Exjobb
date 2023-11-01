from pyexcel_ods3 import save_data
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ranksums
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
"Total Calc Surface Area", #7
"Total Calc Surface/Volume Ratio", #8
"Total Calc Elongation", #9
"Total Calc Flatness", #10
"Total Calc Sphereicity", #11
"Largest Calc Surface Area", #12
"Largest Calc Surface/Volume Ratio", #13
"Largest Calc Elongation", #14
"Largest Calc Flatness", #15
"Largest Calc Sphereicity", #16
"Mean Calc-Lumen Distance", #17
]

def Scatter_plot(feature1,feature2):
    S,AS = seperate_data(list_of_features[feature1])
    S2,AS2 = seperate_data(list_of_features[feature2])
    print(len(S))
    print(len(AS))
    plt.scatter( S, [0] * S.shape[0], c="Red")
    plt.scatter(AS, [1] * AS.shape[0], c="Blue")
    plt.scatter( S, S2, c="Red")
    plt.scatter(AS, AS2, c="Blue")
    plt.show()

def regression(feature1,feature2,classes,file):
    #if feature1 == "Car Score" or feature2 == "Car Score" or feature1 == "Total Calc Sphericity":
    #print("hej")
    file = file[file[feature1].notna()]
    #print(file)
        #file = file.dropna()
    #file.drop
    #print("hoj")
    #print(file)
    sb.lmplot(data=file, x=feature1, y=feature2, hue=classes)
    stat = stats.pearsonr(file[feature1], file[feature2])
    
    #print(stat)
    plt.figtext(0.6, 0.95, "p-value "+ str(stat[1]))
    plt.show()

def distribution(file, feature1):
    S,AS = seperate_data(feature1)
    S = S[np.logical_not(np.isnan(S))]
    AS = AS[np.logical_not(np.isnan(AS))]
    print(ranksums(S, AS))
    print(ranksums(S, AS)[1])
    S = S[np.logical_not(np.isnan(S))]
    sb.displot(file,x=feature1, hue="S/AS",kde=True)
    plt.figtext(0.6, 0.95, "p-value: " + str(ranksums(S, AS)[1]))
    plt.show()

def heatmap(file):
    #plt.figure(figsize=(100,100))
    data = file.iloc[:,3:]
    cormat = data.corr()
    round(cormat,2)
    sb.heatmap(cormat,xticklabels=True, yticklabels=True)
    
    plt.show()

def PCA_SAS(filename):
    
    pca = PCA(n_components=2)
    data = filename
    #data = filename.dropna()
    data_values = data.iloc[:,0:-1]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    #print(data_values)
    data_values = data_values.iloc[:,3:-1]
    #print(y)
    #print(len(y))
    
    X_r = pca.fit(data_values).transform(data_values)
    #print(X_r)
    #print(len(X_r))
    
    print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
    )
    #print(pca.components_)
    df = pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2'])
    #print(pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2']))
    df.to_csv('PCA_SAS.csv')
    plt.figure()
    colors = ["orange", "navy"]
    lw = 2

    for color, i, target_name in zip(colors, ["S", "AS"], ["S", "AS"]):
        plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of dataset")
    plt.figtext(0.01, 0.95, "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_))
    
    plt.show()

def LDA_test(filename):

    lda = LinearDiscriminantAnalysis(n_components=1)
    data = filename
    data_values = data.iloc[:,0:-1]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    #print(data_values)
    data_values = data_values.iloc[:,3:-1]
    lda.fit(data_values, y)
    print(lda.scalings_)
    #print(max(lda.scalings_))
    #print(lda.transform(np.identity(28)))
    X_r2 = lda.transform(data_values)
    df = pd.DataFrame(lda.scalings_.T,columns=data_values.columns,index = ['LDA-1'])
    #print(pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2']))
    df.to_csv('LDA_SAS.csv')
    #print(y)
    #print(X_r2)
    plt.figure()
    colors = ["orange", "navy"]
    #print(X_r2[y=="S"])
    #print(X_r2[y=="AS"])
    # plotting first histogram
    #plt.hist(X_r2[y=="S"], color="orange")
    
# plotting second histogram
    #plt.hist(X_r2[y=="AS"],color="navy")
 
# Showing the plot using plt.show()
    #plt.show()
    for color, i, target_name in zip(colors, ["S", "AS"], ["S", "AS"]):
        plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, ]*0, alpha=0.8, color=color, label=target_name
        )
        
    
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of dataset")
    plt.show()
def PCA_Stenosis(filename):
    pca = PCA(n_components=2)
    data = filename
    #data = filename.dropna()
    
    #print(data_values)
    data_values = data.iloc[:,0:-4]
    #print(data_values)
    data_values = data_values.dropna()
    y = data_values["Stenosis Label"].values
    data_values = data_values.iloc[:,3:-4]
    #print(y)
    #print(len(y))
    #print(type(y[0]))
    X_r = pca.fit(data_values).transform(data_values)
    #print(X_r)
    #print(len(X_r))
    
    print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
    )
    df = pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2'])
    #print(pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2']))
    df.to_csv('PCA_Stenosis.csv')
    colors = ["navy", "turquoise", "red", "green"]
    lw = 2

    for color, i, target_name in zip(colors, ["Light", "Medium", "Severe", "Extreme"], ["Light", "Medium", "Severe", "Extreme"]):
        
        plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of dataset")
    #plt.figtext(0.01, 0.95, "explained variance ratio (first two components): %s"
    #% str(pca.explained_variance_ratio_))
    plt.show()

def LDA_Stenosis(filename):
    lda = LinearDiscriminantAnalysis(n_components=3)
    data = filename
    #data = filename.dropna()
    data_values = data.iloc[:,0:-4]
    #print(data_values)
    data_values = data_values.dropna()
    y = data_values["Stenosis Label"].values
    data_values = data_values.iloc[:,3:-4]
    
    #print(y)
    #print(len(y))
    #print(type(y[0]))
    lda.fit(data_values, y)
    print(lda.scalings_)
    X_r = lda.transform(data_values)
    #print(X_r)
    #print(len(X_r))
    df = pd.DataFrame(lda.scalings_.T,columns=data_values.columns,index = ['LDA-1','LDA-2','LDA-3'])
    #print(pd.DataFrame(pca.components_,columns=data_values.columns,index = ['PC-1','PC-2']))
    df.to_csv('LDA_Stenosis.csv')
    ax = plt.figure().add_subplot(projection='3d')
    #print(pca.components_)
    #plt.figure()
    colors = ["navy", "turquoise", "red", "green"]
    lw = 2

    for color, i, target_name in zip(colors, ["Light", "Medium", "Severe", "Extreme"], ["Light", "Medium", "Severe", "Extreme"]):
        
        ax.scatter(
        X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2],color=color, alpha=0.8, lw=lw, label=target_name
        )

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of dataset")
    #plt.figtext(0.01, 0.95, "explained variance ratio (first two components): %s"
    #% str(pca.explained_variance_ratio_))
    plt.show()
#PCA_SAS(file)
#heatmap(file)
#LDA_test(file)
#Scatter_plot(1,2)
#regression("Total Calc ","Car Score", "S/AS",file)
regression("Total Calc Flatness","Mean Calc-Lumen Distance", "S/AS",file)
#distribution(file,"Total Calc Flatness")
#PCA_Stenosis(file)
#LDA_Stenosis(file)