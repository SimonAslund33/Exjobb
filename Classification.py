from pyexcel_ods3 import save_data
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ranksums
from sklearn.model_selection import train_test_split
#from Feature_Analysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler,NearMiss

def load_file(path):
    xls = pd.read_csv(path)
    #df1 = pd.read_excel(xls, 'file')
    return xls


def LDA_classifier(filename,samplemethod):

    lda = LinearDiscriminantAnalysis(n_components=1)
    data = filename
    data_values = data.iloc[:,0:]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    #print(data_values)
    data_values = data_values.iloc[:,3:]
    X_train,X_test,y_train,y_test=train_test_split(data_values,y,stratify=y,test_size=0.4,random_state=42)
    if samplemethod == "RandomOverSampler":
        print("RandomOverSampler Up Sampling Method Used")
        oversample = RandomOverSampler(sampling_strategy=0.5)
    # fit and apply the transform
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    if samplemethod == "SMOTE":
        print("SMOTE Up Sampling Method Used")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if samplemethod == "RandomUnderSampler":
        print("RandomUnderSampler Down Sampling Method Used")
        undersample = RandomUnderSampler(random_state=42)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
    if samplemethod == "NearMiss":
        print("NearMiss Down Sampling Method Used")
        nearmiss = NearMiss(version=2)
        X_train, y_train = nearmiss.fit_resample(X_train, y_train)
    print(X_train)
    print(y_train)
    lda.fit(X_train, y_train)
    print(lda.scalings_)
    #print(max(lda.scalings_))
    #print(lda.transform(np.identity(28)))
    #X_r2 = lda.transform(X_train)
    #df = pd.DataFrame(lda.scalings_.T,columns=data_values.columns,index = ['LDA-1'])
    #print(lda.predict(X_test))
    #print(y_test)
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(lda, X_test, y_test, scoring='accuracy')
    #print(scores)
    confusion_lda=confusion_matrix(y_test,lda.predict(X_test))
    print(confusion_lda)
    df = pd.DataFrame(confusion_lda.T,columns=['AS','S'],index = ['AS','S'])
    ax = sb.heatmap(df,annot=True,xticklabels=True, yticklabels=True)
    ax.set(xlabel="True labels", ylabel="Predicted labels")
    ax.xaxis.tick_top()
    plt.show()
    # summarize result
    #print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

def knn(filename,samplemethod):
    
    
    
    data = filename
    data_values = data.iloc[:,0:]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    data_values = data_values.iloc[:,3:30]
  
    #data_values = data_values.drop(['Number of calc', 'Largest Calc Vol','Max Calc Arc','Mean Calc Arc','Largest Calc Surface Area','Largest Calc Surface/Vol Ratio','Lesion Calc Proportion','Largest IPH','Number of IPHs','Largest LRNC','Number of LRNCs','Plaque Burden/Vol Ratio','Mean Calc-Lumen Distance','Total Calc Elongation','Total Calc Flatness','Total Calc Sphericity'], axis=1)
    X_train,X_test,y_train,y_test=train_test_split(data_values,y,stratify=y,test_size=0.4,random_state=42)
    if samplemethod == "RandomOverSampler":
        print("RandomOverSampler Up Sampling Method Used")
        oversample = RandomOverSampler(sampling_strategy=0.5)
    # fit and apply the transform
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    if samplemethod == "SMOTE":
        print("SMOTE Up Sampling Method Used")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if samplemethod == "RandomUnderSampler":
        print("RandomUnderSampler Down Sampling Method Used")
        undersample = RandomUnderSampler(random_state=42)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
    if samplemethod == "NearMiss":
        print("NearMiss Down Sampling Method Used")
        nearmiss = NearMiss(version=2)
        X_train, y_train = nearmiss.fit_resample(X_train, y_train)
    print(len(X_train))
    print(len(X_test))
    clf = LinearDiscriminantAnalysis()
    lda= clf.fit_transform(X_train, y_train)
    print(clf.scalings_)
    #print(lda.transform(np.identity(28)))
    #X_r2 = lda.transform(X_train)
    """
    knn_scores=[]

    for k in range(1,20):
        knn=KNeighborsClassifier(n_neighbors=k)
        scores=cross_val_score(knn,lda,y_train,cv=5)
        
        knn_scores.append(scores.mean())
    x_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    x_labels = x_ticks
    """
    #plt.plot([k for k in range(1,20)],knn_scores)
    #plt.xticks(ticks=x_ticks, labels=x_labels)
    #plt.grid()
    
    lda_test=clf.transform(X_test)
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(lda,y_train)
    scores=cross_val_score(knn,lda,y_train,cv=5)
    print(scores)
    confusion_knn=confusion_matrix(y_test,knn.predict(lda_test))
    print(confusion_knn)
    df = pd.DataFrame(confusion_knn.T,columns=['AS','S'],index = ['AS','S'])
    ax = sb.heatmap(df,annot=True,xticklabels=True, yticklabels=True)
    ax.set(xlabel="True labels", ylabel="Predicted labels")
    ax.xaxis.tick_top()
    plt.show()
def regression_model(file):
    """
    data = file
    data_values = data.iloc[:,0:]
    #print(data_values) 
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    #for i in range(len(y)):
    #    if y[i] == "S":
    #        y[i] = 1
    #    if y[i] == "AS":
    #        y[i] = 0
    #print(y)
    X  = data_values.iloc[:,3:27]
    print(X)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    print(lda.scalings_)
    X = lda.transform(X)
    #split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

    #instantiate the model
    #log_regression = LogisticRegression()

    #fit the model using the training data
    #log_regression.fit(X_train,y_train)
    #define metrics
    y_pred_proba = log_regression.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label="S")

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    """
    data = file
    data_values = data.iloc[:,0:]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    data_values = data_values.iloc[:,3:27]

    X_train,X_test,y_train,y_test=train_test_split(data_values,y,stratify=y,test_size=0.4,random_state=42)
    print(len(X_train))
    print(len(X_test))
    clf = LinearDiscriminantAnalysis()
    lda= clf.fit_transform(X_train, y_train)
    print(clf.scalings_)
    #print(lda.transform(np.identity(28)))
    #X_r2 = lda.transform(X_train)
    
    
    lda_test=clf.transform(X_test)
    knn=KNeighborsClassifier(n_neighbors=7)
    knn.fit(lda,y_train)
    y_scores = knn.predict_proba(lda_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_scores[:, 1],pos_label="S")
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of kNN')
    plt.show()
def logregression_plot(file):
    data = file
    data_values = data.iloc[:,0:]
    #print(data_values) 
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    #for i in range(len(y)):
    #    if y[i] == "S":
    #        y[i] = 1
    #    if y[i] == "AS":
    #        y[i] = 0
    #define the predictor variable and the response variable
    x = data_values.iloc[:,3:27]
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(x, y)
    print(lda.scalings_)
    x = lda.transform(x)
    #y = data['default']

#plot logistic regression curve
    sb.regplot(x=x, y=y, data=data,logistic=True, ci=None)
def gradientboostingclassifier(filename,samplemethod):
    data = filename
    data_values = data.iloc[:,0:]
    data_values = data_values.dropna()
    y = data_values["S/AS"].values
    data_values = data_values.iloc[:,3:30]
  
    #data_values = data_values.drop(['Number of calc', 'Largest Calc Vol','Max Calc Arc','Mean Calc Arc','Largest Calc Surface Area','Largest Calc Surface/Vol Ratio','Lesion Calc Proportion','Largest IPH','Number of IPHs','Largest LRNC','Number of LRNCs','Plaque Burden/Vol Ratio','Mean Calc-Lumen Distance','Total Calc Elongation','Total Calc Flatness','Total Calc Sphericity'], axis=1)
    X_train,X_test,y_train,y_test=train_test_split(data_values,y,stratify=y,test_size=0.4,random_state=42)
    if samplemethod == "RandomOverSampler":
        print("RandomOverSampler Up Sampling Method Used")
        oversample = RandomOverSampler(sampling_strategy=1)
    # fit and apply the transform
        X_train, y_train = oversample.fit_resample(X_train, y_train)
    if samplemethod == "SMOTE":
        print("SMOTE Up Sampling Method Used")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if samplemethod == "RandomUnderSampler":
        print("RandomUnderSampler Down Sampling Method Used")
        undersample = RandomUnderSampler(random_state=42)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
    if samplemethod == "NearMiss":
        print("NearMiss Down Sampling Method Used")
        nearmiss = NearMiss(version=2)
        X_train, y_train = nearmiss.fit_resample(X_train, y_train)
    gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
    gradient_booster.fit(X_train,y_train)
    print(metrics.classification_report(y_test,gradient_booster.predict(X_test)))
file = load_file(r"E:\file.csv")

#knn(file,"NearMiss")
#LDA_classifier(file,"RandomOverSampler")
#regression_model(file)
#logregression_plot(file)
gradientboostingclassifier(file, "SMOTE")