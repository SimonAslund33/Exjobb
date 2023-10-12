from pyexcel_ods3 import save_data
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
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
#print(file["Total Calcification Volume"])
#plt.scatter( X, [0] * X.shape[0])
S,AS = seperate_data(list_of_features[1])
S2,AS2 = seperate_data(list_of_features[2])
#print(hej,hej2)
"""
plt.scatter( S, [0] * S.shape[0], c="Red")
plt.scatter(AS, [1] * AS.shape[0], c="Blue")
plt.scatter( S, S2, c="Red")
plt.scatter(AS, AS2, c="Blue")


print(S.shape)
print(AS.shape)
"""
"""
dataset = pd.DataFrame({
    "value": np.concatenate((S, AS)),
    "type": np.concatenate((np.ones(S.shape), np.zeros(AS.shape)))
})
dataset.info()
sb.violinplot(x="type", y="value", data=dataset)
sb.swarmplot(x="type", y="value", data=dataset, size=2, color="k", alpha=0.3)
"""

#sb.lmplot(data=file, x="Number of calcifications", y="Total Calcification Volume", hue="S/AS")
#sb.displot(data=file, x="Number of calcifications", row= "S", kde=True)
plt.show()