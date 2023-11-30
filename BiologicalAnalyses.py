
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
import statsmodels.stats.multitest as mc
def load_file(path):
    #xls = pd.read_table(path)
    xls = pd.read_table(path, delim_whitespace=True)
    #print(xls)
    #df1 = pd.read_excel(xls, 'file')
    return xls



def biological_analysis(file):


    
    #data = filename.dropna()
    #data_values = data.iloc[:,0:-1]
    #data_values = data_values.dropna()
    #file = pd.DataFrame(file)
   # data = pd.read_table(file, delim_whitespace=True)
    #print(file)
    #file = (file[file["fc"]])
    X = file["fc"]
    y = file["p"]
    y_fdr = mc.fdrcorrection(y, alpha=0.05, method='indep', is_sorted=False)
    #print(sorted(y_fdr[1])[0:20])
    #print(sorted(y_fdr[1])[-20:-1])
    #labeledfiles = (file["p"]< 0.001)
    #labels = labeledfiles["gene"]
    for i in range(len(file["p"])-1):
        i = i+1
        if ((-1)*np.log10(file.iloc[i,2])) < 2.7 and abs((np.log2(file.iloc[i,3]))) < 5:
            file.iloc[i,1] = None

    #a = file[["p"] > 0.001]
    #a['gene'] = None
    y = (-1)*(np.log10(y))
    X = np.log2(X)
    #file["fc"]< 100)
    print(max(X))
    print(np.where(X == max(X)))
    plt.figure()
    #binwidth = 0.05
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.title('fc Flatness')
    #ax1.hist(y, range=[0,1],bins=20)
    #ax2.hist(y_fdr[1], range=[0,1], bins=20)
    #colors = ["orange", "navy"]
    #lw = 2
    #plt.hist(y_fdr[1], range=[0,1])
    #plt.hist(y_fdr[1], range=[0,1])
    # plotting second histogram
    #plt.hist(X_r2[y=="AS"],color="navy")
    labels = file["gene"]
    
    plt.scatter(X,y,s=0.3,)
    for X, y, s in zip(X, y, labels):
        plt.text(X, y, s, fontsize=7)
    
    

    #plt.legend(loc="best", shadow=False, scatterpoints=1)
    #plt.title("PCA of dataset")
    #plt.figtext(0.01, 0.95, "explained variance ratio (first two components): %s"
    plt.show()

file_Flat = load_file(r"C:\Users\simon\Downloads\Top15Bot15FlatnessBio.txt")
file_Elong = load_file(r"C:\Users\simon\Downloads\Top15Bot15ElongationBio.txt")
file_Spher = load_file(r"C:\Users\simon\Downloads\Top15Bot15SphericityBio.txt")
biological_analysis(file_Flat)