import numpy as np
import napari
import pandas as pd
#import xlwings as xw
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import cc3d
from radiomics import base, cShape, deprecated, featureextractor
import SimpleITK as sitk
import six
import os
import glob
import csv
from pyexcel_ods3 import save_data

Patients = os.listdir(r"E:\Controlled_Patients")

#slice = np.load(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Slices.txt.npy')
#unwraps = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Unwraps.txt.npy")
#plaque_volume = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Plaque_volume.txt.npy")

xls = pd.ExcelFile(r"C:\Users\simon\CarScore.xlsx")
df2 = pd.read_excel(xls, 'Alla med CAR-score')
#print(df2.iloc[:,0])
#E:\Controlled_Patients\BiKE_0623\wi-07F88BAA\RightCarotid\subvol.nrrd
xls = pd.ExcelFile(r'C:\Users\simon\Downloads\BiKE_imported_csv.xlsx')
df1 = pd.read_excel(xls, 'BiKE Elucid plus operation')
def create_path(patient):
    for i in range(len(df1)):
        #print(df1.iloc[i,1])
        if df1.iloc[i,1] == patient:
            #print("found!")
            s = df1.iloc[i,12]
            Symptom = df1.iloc[i,13]
            break
    side = s + r"Carotid"
    slicepath = glob.glob(r"E:\Controlled_Patients/" + patient + r"/*/" + side + r"/Slices.txt.npy")
    unwrappath = glob.glob(r"E:\Controlled_Patients/" + patient + r"/*/" + side + r"/Unwraps.txt.npy")
    volumepath = glob.glob(r"E:\Controlled_Patients/" + patient + r"/*/" + side + r"/Plaque_volume.txt.npy")
    slicepath = np.load(slicepath[0])
    unwrappath = np.load(unwrappath[0])
    volumepath = np.load(volumepath[0])
    return slicepath,unwrappath,volumepath,Symptom

def number_of_calcifications(volume):

    #labels_out = cc3d.connected_components(volume)
    labels_out, N = cc3d.connected_components(volume, return_N=True)
    stats = cc3d.statistics(labels_out)
    if len(stats['voxel_counts']) > 1:
        largest_calcification = max(stats['voxel_counts'][1:])
        idx = np.where(stats['voxel_counts'][1:]==largest_calcification)
        mean_size_calcification = sum(stats['voxel_counts'][1:])/N
        largest_calc_vol = only_calcifications(labels_out,idx)
    else:
        largest_calcification = 0
        mean_size_calcification = 0
        largest_calc_vol = np.zeros_like(labels_out)

    return labels_out,largest_calcification,mean_size_calcification,N,largest_calc_vol

def only_calcifications(volume,idx):
    empty = np.zeros_like(volume)
    if np.where(volume == 1):
        Calc_idx = np.where(volume == idx)
    
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
        if angles[0] > 0:
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
    if arc_list == []:
        return 0
    else:
        return sum(max(arc_list))
def Calc_mean_arc(arc_list):
    if arc_list == []:
        return 0
    else:
        tot_mean = []
        for i in arc_list:
            mean = sum(i)/len(i)
            if mean > 0:
                tot_mean.append(mean)
    
        return (sum(tot_mean)/len(tot_mean))

def Calc_Area_calculations(Calc_volume):
    if 1 in Calc_volume:
        hej = sitk.GetImageFromArray(Calc_volume)
        extractor= featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute(hej,hej)
    #print(type(result))
        features = []
        for key,value in result.items():
        #print ("\t",key,":",value)
            if key == "original_shape_Elongation":
                features.append(value)
            elif key == "original_shape_Flatness":
                features.append(value)
            elif key == "original_shape_Sphericity":
                features.append(value)
            elif key == "original_shape_SurfaceArea":
                features.append(value)
            elif key == "original_shape_SurfaceVolumeRatio":
                features.append(value)
            elif key == "original_shape_VoxelVolume":
                features.append(value)
            else:
                continue
        return features[0],features[1],features[2],features[3],features[4],features[5]
    else:
        return 0,0,0,0,0,0
def Calc_Lumen_Distance(unwraps):
    mean_tot = []
    for j in range(unwraps.shape[2]):
        calc_pixels = np.where(unwraps[:,:,j] == 1)
        if calc_pixels[0].size > 0:
           
            unique_angles = np.unique(calc_pixels[0])
            distances = []
            for i in unique_angles:
                C = max(np.where(unwraps[i,:] == 1)[0])
                L = max(np.where(unwraps[i,:] == 5)[0])
                distances.append(L-C)
            mean = sum(distances)/len(distances)
            mean_tot.append(mean)
    if mean_tot == []:
        return None
    else:
        mean_tot = sum(mean_tot)/len(mean_tot)
        return mean_tot
    
def carSCore(Patient):
    #print(int(Patient[6:]))
    #print(df2["BiKE-ID"][0:])
    for i in range(len(df2.iloc[:,0])):
        #print(Patient[6:])
        #print(i)
        if int(Patient[6:]) == df2.iloc[i,0]:
            data = {df2.iloc[i,1]}
            data = int(list(data)[0])
            if data < 1:
                score = data*100
            else:
                score = data
        else:
            score = None
    return score
def main():

    Dict = {}

    for i in Patients:
        print(i)
        slice,unwraps,plaque_volume,Symptom = create_path(i)
        arcs = Arc_calculations(unwraps)
        max_arc = Calc_maximum_arc(arcs)
        mean_arc = Calc_mean_arc(arcs)
        calc_volume = only_calcifications(plaque_volume,1)
        labels_out,largest_calcification,mean_size_calcification,N_of_calc,idx = number_of_calcifications(calc_volume)
        #largest_calcification = max(stats['voxel_counts'][1:])
        #mean_size_calcification = sum(stats['voxel_counts'][1:])/N_of_calc
        Elongation,Flatness,Sphericity,SurfaceArea,SurfaceVolumeRatio,TotalCalcVolume = Calc_Area_calculations(calc_volume)
        LargeElongation,LargeFlatness,LargeSphericity,LargeSurfaceArea,LargeSurfaceVolumeRatio,LargeTotalCalcVolume = Calc_Area_calculations(labels_out)
        mean_CalcLumenDistance = Calc_Lumen_Distance(unwraps)

        thisdict = {
  
  "S/AS": Symptom,
  "Total Calcification Volume": TotalCalcVolume,
  "Number of calcifications": N_of_calc,
  "Largest Calcification Volume": largest_calcification,
  "Mean Calcification volume": mean_size_calcification,
  "Maximum Calcification Arc": max_arc,
  "Mean Calcification Arc": mean_arc,
  "Total Calcification Surface Area": SurfaceArea,
  "Total Calcification Surface/Volume Ratio": SurfaceVolumeRatio,
  "Total Calcification Elongation": Elongation,
  "Total Calcification Flatness": Flatness,
  "Total Calcification Sphereicity": Sphericity,
  "Largest Calcification Surface Area": LargeSurfaceArea,
  "Largest Calcification Surface/Volume Ratio": LargeSurfaceVolumeRatio,
  "Largest Calcification Elongation": LargeElongation,
  "Largest Calcification Flatness": LargeFlatness,
  "Largest Calcification Sphereicity": LargeSphericity,
  "Mean Calcification-Lumen Distance": mean_CalcLumenDistance


}   
        Dict[i] = thisdict
    
    #print(Dict)
    df = pd.DataFrame(Dict).T  # transpose to look just like the sheet above
    df.to_csv('file.csv')
    #df.to_excel('file.xls')
    #print(thisdict)
    #with napari.gui_qt():
    #    viewer = napari.Viewer()
    #    viewer.add_image(labels_out)
    #napari.run()
for i in Patients[:-2]:
    print(carSCore(i))  

#print(df2.iloc[:,0])
#print("Mean Calcification Distance to Lumen:", mean_CalcLumenDistance)
#print("Maximum Calcium Arc:", max_arc)
#print(Area_calc)
#print(N_of_calc)
#print(largest_calcification)
#print(mean_size_calcification)

    #print(key == "original_shape_Flatness")


#main()
"""
hej = Patients[119]
slice,unwraps,plaque_volume,Symptom = create_path(hej)
print(hej)
#calc_volume = only_calcifications(plaque_volume)
#labels_out,largest_calcification,mean_size_calcification,N_of_calc = number_of_calcifications(calc_volume)
#arcs = Arc_calculations(unwraps)
#print(cc3d.statistics(labels_out))
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(plaque_volume)
#    viewer.add_image(unwraps)
#    viewer.add_image(slice)
napari.run()
"""