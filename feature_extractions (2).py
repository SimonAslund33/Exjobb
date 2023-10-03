import numpy as np
import napari
import pandas as pd
#import xlwings as xw
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import cc3d

slice = np.load(r'E:\BiKE_0846\wi-266BCEEA\RightCarotid\Slices.txt.npy')
unwraps = np.load(r"E:\BiKE_0846\wi-266BCEEA\RightCarotid\Unwraps.txt.npy")
plaque_volume = np.load(r"E:\BiKE_0846\wi-266BCEEA\RightCarotid\Plaque_volume.txt.npy")


xls = pd.ExcelFile(r'C:\Users\simon\Downloads\BiKE_imported_csv.xlsx')
df1 = pd.read_excel(xls, 'BiKE Elucid plus operation')
def number_of_calcifications(volume):

    #labels_out = cc3d.connected_components(volume)
    labels_out, N = cc3d.connected_components(volume, return_N=True)
    stats = cc3d.statistics(labels_out)
    return labels_out,stats,N

def only_calcifications(volume):
    empty = np.zeros_like(volume)
    Calc_idx = np.where(volume == 1)
    for i in range(len(Calc_idx[0])):
        empty[Calc_idx[0][i],Calc_idx[1][i],Calc_idx[2][i] ] = 1

    return empty

def Arc_calculations(unwrap):
    angles_tot = []
    #print(angles_tot)
    for j in range(unwrap.shape[2]):
    #print(unwrap.shape)
        calc_pixels = np.where(unwrap[:,:,j] == 1)
    #print(calc_pixels)
        unique_angles = np.unique(calc_pixels[0])
        #print(unique_angles)
        angles = []
    #idx = 1
        start_idx = 1
        for i in range(1,len(unique_angles)): 
            #print(unique_angles)
            if unique_angles[i] == 359 and unique_angles[0] == 0:
                #print("end")
                
                start_idx = start_idx-angles[0]
                angles.remove(angles[0])
                break
            elif unique_angles[i] == unique_angles[i-1] + 1:
                continue
         
            else: 
                #print(i)   
                angles.append((i+1) - start_idx)
                start_idx = i+1
            #print(start_idx)
    
        angles.append(len(unique_angles)+1 - start_idx)
        angles_tot.append(angles)

    
    return angles_tot

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

def Calc_maximum_arc(arc_list):
    return max(arc_list)  
def Calc_mean_arc(arc_list):
    tot_mean = []
    for i in arc_list:
        mean = sum(i)/len(i)
        if mean > 0:
            tot_mean.append(mean)
    
    return (sum(tot_mean)/len(tot_mean))

    
arcs = Arc_calculations(unwraps)
max_arc = Calc_maximum_arc(arcs)
mean_arc = Calc_mean_arc(arcs)
calc_volume = only_calcifications(plaque_volume)
labels_out,stats,N_of_calc = number_of_calcifications(calc_volume)
largest_calcification = max(stats['voxel_counts'][1:])
mean_size_calcification = sum(stats['voxel_counts'][1:])/N_of_calc
print(N_of_calc)
print(largest_calcification)
print(mean_size_calcification)
with napari.gui_qt():
    viewer = napari.Viewer()
#    viewer.add_image(plaque_volume)
#    viewer.add_image(slice)
    #viewer.add_image(labelVolume)
    #viewer.add_image(unwraps)
   # viewer.add_image(calc_volume)
    viewer.add_image(labels_out)
napari.run()
