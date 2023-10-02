import numpy as np
import napari
import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

slice = np.load(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Slices.txt.npy')
unwraps = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Unwraps.txt.npy")
plaque_volume = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Plaque_volume.txt.npy")


xls = pd.ExcelFile(r'C:\Users\simon\Downloads\BiKE_imported_csv.xlsx')
df1 = pd.read_excel(xls, 'BiKE Elucid plus operation')

def new_center(image):
    centers = []
    for j in range(image.shape[2]):
        thresh = cv2.inRange(image[:,:,j], 6, 6)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        CL = np.where(image[:,:,j] == 7)
        print(contours)
        distance = 1000
        for i in contours:
            point = (sum(i)/len(i))
            print(point)
            point = [round(j) for j in point[0]]
            dist = np.sqrt( (point[1] - CL[0])**2 + (point[0] - CL[1])**2 )
            print(dist)
            if dist < distance:
                distance = dist
                closest_point = point
            
        centers.append(closest_point)
        
    return centers

PatientID = []
CalcVols = []
symptoms = []

def calcvol():
    for i in range(len(df1)):
        if df1.iloc[i,5] == "LeftCarotid" and df1.iloc[i,12] == "Left" and df1.iloc[i,14] == "Bifurcation":
            patient = df1.iloc[i,1]
            CV1 = df1.iloc[i,16]
            CV2 = df1.iloc[i+1,16]
            CV = CV1+CV2
            PatientID.append(patient)
            CalcVols.append(CV)
            symptoms.append(df1.iloc[i,13])
            

        if df1.iloc[i,5] == "RightCarotid" and df1.iloc[i,12] == "Right" and df1.iloc[i,14] == "Bifurcation":
            patient = df1.iloc[i,1]
            CV1 = df1.iloc[i,16]
            CV2 = df1.iloc[i+1,16]
            CV = CV1+CV2
            PatientID.append(patient)
            CalcVols.append(CV)
            symptoms.append(df1.iloc[i,13])
        
        else:
            continue
        
    indicesS = [i for i in range(len(symptoms)) if symptoms[i] == "S"]
    indicesAS = [i for i in range(len(symptoms)) if symptoms[i] == "AS"]

    symptom_vols = []
    Asymptom_vols = []
    for i in indicesS:
        symptom_vols.append(CalcVols[i])

    for i in indicesAS:
        Asymptom_vols.append(CalcVols[i])


    fig, ax = plt.subplots()
    sns.distplot(symptom_vols, hist=False, rug=True,label="Symptomatic",color="red")
    sns.distplot(Asymptom_vols, hist=False, rug=True,label="Asymptomatic",color="skyblue")
    ax.set_xlim(0,1500 )
    ax.set_xlabel( "CalcVol")
    ax.set_ylabel( "count")
    plt.show()
cent = new_center(slice)
    
#print(cent)  
for i in range(slice.shape[2]):
    #print(cent[i])
    slice[cent[i][1],cent[i][0],i] = 8
    
with napari.gui_qt():
    viewer = napari.Viewer()
    #viewer.add_image(total_unwrap)
    viewer.add_image(slice)
    #viewer.add_image(labelVolume)
    #viewer.add_image(unwraps)
    #viewer.add_image(cent)
napari.run()
