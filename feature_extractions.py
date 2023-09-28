import numpy as np
import napari
import pandas as pd

slice = np.load(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Slices.txt.npy')
unwraps = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Unwraps.txt.npy")
plaque_volume = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Plaque_volume.txt.npy")

print(slice.shape)
print(unwraps.shape)
print(plaque_volume.shape)

dataframe1 = pd.read_excel(r'C:\Users\simon\Downloads\BiKE_imported_csv.xlsx')

#print(dataframe1["location"])


rows = len(dataframe1["subjectID"])
#cols = len(dataframe1)
print(rows)
Calc_vols = []

for i in range(rows):
    a = dataframe1.iloc[i,11]
    a = str(a)
    if a[0] == "P":
        print(i)
        CV = float(dataframe1.iloc[i,12])
        CV2 = float(dataframe1.iloc[i+1,12])
        CV3 = CV + CV2
        Calc_vols.append(CV3)
        print(CV,CV2,CV3)

        
    
    

