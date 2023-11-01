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
#from Nrrd_test import create_file_paths, create_volume, Find_center, build_Center_Line
import math
from pyevtk.hl import gridToVTK
Patients = os.listdir(r"E:\Controlled_Patients")

#slice = np.load(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Slices.txt.npy')
#unwraps = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Unwraps.txt.npy")
#plaque_volume = np.load(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\Plaque_volume.txt.npy")

xls = pd.ExcelFile(r"E:\CarScore.xlsx")
df2 = pd.read_excel(xls, 'Alla med CAR-score')
#print(df2.iloc[:,0])
#E:\Controlled_Patients\BiKE_0623\wi-07F88BAA\RightCarotid\subvol.nrrd
xls = pd.ExcelFile(r'E:\BiKE_imported_csv.xlsx')
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
    """
    shape = volumepath.shape
    w = shape[0]
    h = shape[1]
    noSlices = shape[2]
    x = np.arange(0, w+1, dtype=np.int32)
    y = np.arange(0, h+1, dtype=np.int32)
    z = np.arange(0, noSlices+1, dtype=np.int32)
    hej = r"E:\Controlled_Patients/" + patient + r"/*/" + side + r"/Plaque_volume"
    print(hej)
    gridToVTK(hej, x, y, z, cellData = {'Plaque_volume': volumepath})
    """
    return slicepath,unwrappath,volumepath,Symptom

def number_of_calcifications(volume):

    #labels_out = cc3d.connected_components(volume)
    labels_out, N = cc3d.connected_components(volume, return_N=True)
    stats = cc3d.statistics(labels_out)
    if len(stats['voxel_counts']) > 1:
        Total_volume = sum(stats['voxel_counts'][1:])
        largest_calcification = max(stats['voxel_counts'][1:])
        idx = np.where(stats['voxel_counts'][1:]==largest_calcification)
        mean_size_calcification = sum(stats['voxel_counts'][1:])/N
        largest_calc_vol = only_calcifications(labels_out,idx)
    else:
        Total_volume = 0
        largest_calcification = 0
        mean_size_calcification = 0
        largest_calc_vol = np.zeros_like(labels_out)

    return labels_out,largest_calcification,mean_size_calcification,N,largest_calc_vol,Total_volume

def only_calcifications(volume,idx):
    empty = np.zeros_like(volume)
    if np.where(volume == idx):
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
            #print(j)
            if len(unique_angles) == 360:
                angles_tot.append([360])
                break
            elif unique_angles[i] == 359 and unique_angles[0] == 0:
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
        labels_out, N = cc3d.connected_components(Calc_volume, return_N=True)
        labels_out = cc3d.dust(labels_out, threshold=4,connectivity=26, in_place=False)
        stats = cc3d.statistics(labels_out)
        #print(stats)
        Total_volume = sum(stats['voxel_counts'][1:])
        #print(Total_volume)
        Elongation = []
        Flatness = []
        Sphericity = []
        Surface_Area = []
        Surface_Volume_Ratio = []
        Voxel_Volume = []
        compactness = []
        MajorAxis = []
        content = np.unique(labels_out)[1:]
        for i in content:
            volume = only_calcifications(labels_out,i)
            stats = cc3d.statistics(volume)
        
            Tot_volume = sum(stats['voxel_counts'][1:])
            weight = Tot_volume/Total_volume
            #print(weight)
            hej = sitk.GetImageFromArray(volume)
            extractor= featureextractor.RadiomicsFeatureExtractor()
            result = extractor.execute(hej,hej)
            print(result)
    #print(type(result))
            features = []
            for key,value in result.items():
        #print ("\t",key,":",value)
                if key == "original_shape_Elongation":
                    Elongation.append((1-value)*weight)
                elif key == "original_shape_Flatness":
                    Flatness.append((1-value)*weight)
                elif key == "original_shape_Sphericity":
                    Sphericity.append(value*weight)
                elif key == "original_shape_SurfaceArea":
                    Surface_Area.append(value)
                elif key == "original_shape_SurfaceVolumeRatio":
                    Surface_Volume_Ratio.append(value)
                elif key == "original_shape_VoxelVolume":
                    Voxel_Volume.append(value)
                #elif key == 'original_shape_MajorAxisLength':
                #    MajorAxis.append(value)
                #elif key == 'original_shape_compactness_1':
                #    compactness.append(value)
                else:
                    continue
        #print(Elongation)
        #print(Flatness)
        #print(Sphericity)
        if len(Voxel_Volume) > 0:
            #total_volume = sum(Voxel_Volume) 
            #Mean_Elongation = sum(Elongation)/len(Elongation)
            #Mean_Flatness = sum(Flatness)/len(Flatness)
            #Mean_Sphericity = sum(Sphericity)/len(Sphericity)
            Mean_Surface_Area = sum(Surface_Area)/len(Surface_Area)
            Mean_surface_Volume_Ratio = sum(Surface_Volume_Ratio)/len(Surface_Volume_Ratio)
            #Mean_Voxel_Volume = sum(Voxel_Volume)/len(Voxel_Volume)

            max_calc_idx = Voxel_Volume.index(max(Voxel_Volume))
            #Largest_Elongation = Elongation[max_calc_idx]
            #Largest_Flatness = Flatness[max_calc_idx]
            #Largest_Sphericity = Sphericity[max_calc_idx]
            Largest_Surface_Area = Surface_Area[max_calc_idx]
            Largest_Surface_Volume_Ratio = Surface_Volume_Ratio[max_calc_idx]
            #Largest_Voxel_Volume = Voxel_Volume[max_calc_idx] 
            #return 1-Mean_Elongation,1-Mean_Flatness,Mean_Sphericity,Mean_Surface_Area,Mean_surface_Volume_Ratio,Mean_Voxel_Volume,1-Largest_Elongation,1-Largest_Flatness,Largest_Sphericity,Largest_Surface_Area,Largest_Surface_Volume_Ratio,Largest_Voxel_Volume
            return sum(Elongation),sum(Flatness),sum(Sphericity),Mean_Surface_Area,Mean_surface_Volume_Ratio,Largest_Surface_Area,Largest_Surface_Volume_Ratio
        else:
            return None,None,None,None,None,None,None
            #return 0,0,0

    else:
        return None,None,None,None,None,None,None
        #return 0,0,0
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
        #print(Patient)
        #print(i)
        if int(Patient[6:]) == df2.iloc[i,0]:
            #print("score")
            data = {df2.iloc[i,1]}
            
            data = float(list(data)[0])
            #print(data)
            if data < 1:
                #print("hej")
                return round(data*100)
            else:
                #print("hoj")
                return round(data)
        else:
            continue
    return None
def PlaqueBurden(Patient):
    for i in range(len(df1.iloc[:,1])):
        #print(Patient)
        data3 = None
        if Patient == df1.iloc[i,1] and df1.iloc[i,12] == "Right" and df1.iloc[i,5] == "RightCarotid" and df1.iloc[i,14] == "Bifurcation":
            #print(i)
            #print("found right")
            data1 = df1.iloc[i,67]
            data2 = df1.iloc[i+1,67]
            data3 = (data1+data2)/2
            break
        elif Patient == df1.iloc[i,1] and df1.iloc[i,12] == "Left" and df1.iloc[i,5] == "LeftCarotid" and df1.iloc[i,14] == "Bifurcation":
            #print(i)
            #print("found left")
            
            data1 = df1.iloc[i,67]
            #print(data1)
            data2 = df1.iloc[i+1,67]
            #print(data2)
            data3 = (data1+data2)/2
            #print(data3)
            break
        else:
            
            continue
    #print(df1.iloc[478,1], df1.iloc[478,12],df1.iloc[478,5],df1.iloc[478,14])
    
    return data3      

def mean_Stenosis(slice):
    
    for j in range(slice.shape[2]):
        if len(np.where(slice[:,:,j] == 5)[0]) > 0:
            x = np.where(slice[:,:,i] == 6)
        y = np.count_nonzero(slice[:,:,i])
        x = len(x[0])
        sten = x/y
        

def max_stenosis_grade(slice):
    stenosis = []
    for i in range(slice.shape[2]):
        x = np.where(slice[:,:,i] == 6)
        y = np.count_nonzero(slice[:,:,i])
        x = len(x[0])
        sten = x/y
        if sten < 1:
            stenosis.append(1- sten)
    
    if max(stenosis) < 0.60:
        stenosis_label="Light"
    if 0.80 > max(stenosis) > 0.60:
        stenosis_label="Medium"
    if 0.95 > max(stenosis) > 0.80:
        stenosis_label="Severe"
    if max(stenosis) > 0.95:
        stenosis_label="Extreme"

    return(max(stenosis)), stenosis_label, (sum(stenosis)/len(stenosis))

def main():

    Dict = {}

    for i in Patients[2:-2]:
        print(i)
        slice,unwraps,plaque_volume,Symptom = create_path(i)
        arcs = Arc_calculations(unwraps)
        max_arc = Calc_maximum_arc(arcs)
        mean_arc = Calc_mean_arc(arcs)
        calc_volume = only_calcifications(plaque_volume,1)
        labels_out,largest_calcification,mean_size_calcification,N_of_calc,idx,TotalCalcVolume = number_of_calcifications(calc_volume)
        #largest_calcification = max(stats['voxel_counts'][1:])
        #mean_size_calcification = sum(stats['voxel_counts'][1:])/N_of_calc
        Elongation,Flatness,Sphericity,SurfaceArea,SurfaceVolumeRatio,LargeSurfaceArea,LargeSurfaceVolumeRatio = Calc_Area_calculations(calc_volume)
        mean_CalcLumenDistance = Calc_Lumen_Distance(unwraps)
        Car_Score = carSCore(i)
        PlaqueBurden_Vol_Ratio = PlaqueBurden(i)
        #Maximum_Stenosis,mean_stenosis = Max_Stenosis(slice)
        Stenosis_grade,Stenosis_label,mean_stenosis = max_stenosis_grade(slice)
        IPH_volume = only_calcifications(plaque_volume,3)
        labels_out,largest_IPH,mean_size_IPH,N_of_IPH,idx,TotalIPHVolume = number_of_calcifications(IPH_volume)
        #Elong,Flat,Spher,Surf,SurfVolRatio,MeanIPHVolume,LargeElon,LargeFlat,LargeSpher,LargeSurf,LargeSurfVolRatio,LargeTotalCalcVolume = Calc_Area_calculations(IPH_volume)
        LRNC_volume = only_calcifications(plaque_volume,2)
        labels_out,largest_LRNC,mean_size_LRNC,N_of_LRNC,idx,TotalLRNCVolume = number_of_calcifications(LRNC_volume)
        #Elong2,Flat2,Spher2,Surf2,SurfVolRatio2,MeanLRNCVolume,LargeElon2,LargeFlat2,LargeSpher2,LargeSurf2,LargeSurfVolRatio2,LargeTotalCalcVolume = Calc_Area_calculations(IPH_volume)
        thisdict = {
  
  "S/AS": Symptom,
  "Stenosis Label": Stenosis_label,
  "Total Calc Vol": TotalCalcVolume,
  "Number of calc": N_of_calc,
  "Largest Calc Vol": largest_calcification,
  "Mean Calc Vol": mean_size_calcification,
  "Max Calc Arc": max_arc,
  "Mean Calc Arc": mean_arc,
  "Total Calc Surface Area": SurfaceArea,
  "Total Calc Surface/Vol Ratio": SurfaceVolumeRatio,
  "Total Calc Elongation": Elongation,
  "Total Calc Flatness": Flatness,
  "Total Calc Sphericity": Sphericity,
  "Largest Calc Surface Area": LargeSurfaceArea,
  "Largest Calc Surface/Vol Ratio": LargeSurfaceVolumeRatio,
  #"Largest Calc Elongation": LargeElongation,
  #"Largest Calc Flatness": LargeFlatness,
  #"Largest Calc Sphereicity": LargeSphericity,
  "Total IPH Vol": TotalIPHVolume,
  "Largest IPH": largest_IPH,
  "Mean IPH Vol": mean_size_IPH,
  "Number of IPHs": N_of_IPH,
  "Total LRNC Vol": TotalLRNCVolume,
  "Largest LRNC": largest_LRNC,
  "Mean LRNC Vol": mean_size_LRNC,
  "Number of LRNCs": N_of_LRNC,
  "Total Calc/IPH/LRNC Vol": TotalCalcVolume+TotalIPHVolume+TotalLRNCVolume,
  "Plaque Burden/Vol Ratio": PlaqueBurden_Vol_Ratio,
  "Mean Stenosis": mean_stenosis,
  "Stenosis grade": Stenosis_grade,
  "Mean Calc-Lumen Distance": mean_CalcLumenDistance,
  "Car Score": Car_Score
  
      

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
#for i in Patients[4:-2]:
#    print(carSCore(i))  

#import pydicom
#from pydicom.data import get_testdata_files
#list = np.empty([512,512])
#for filename in os.listdir(r"C:\Users\Simon\Downloads\BiKE_0846\BiKE_0846\2195652772\2195653733"):
#    dcm_data = pydicom.dcmread(r"C:\Users\Simon\Downloads\BiKE_0846\BiKE_0846\2195652772\2195653733/" + filename)
#    im = dcm_data.pixel_array
#    list = np.append(list, im, axis=2)

#print(list.shape)

#plt.imshow(im, cmap='gray')
#plt.axis('off')
#plt.title('Axial Slice of a Chest-CT')
#plt.show()
#print(dcm_data)
#ds = pydicom.dcmread(filename)
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
#main()
hej = "BiKE_0788"
slice,unwraps,plaque_volume,Symptom = create_path(hej)
#shape = plaque_volume.shape
#w = shape[0]
#h = shape[1]
#noSlices = shape[2]
#x = np.arange(0, w+1, dtype=np.int32)
#y = np.arange(0, h+1, dtype=np.int32)
#z = np.arange(0, noSlices+1, dtype=np.int32)
#hej = r"E:\Controlled_Patients/" + patient + r"/*/" + side + r"/Plaque_volume"
#gridToVTK("E:\Controlled_Patients\BiKE_0788\wi-1B16AFB4\RightCarotid\Plaque_volume", x, y, z, cellData = {'Plaque_volume': plaque_volume})
#Maximum_Stenosis,mean_stenosis = Max_Stenosis(slice)
#print(hej)
#PlaqueBurden(hej)
#calc_volume = only_calcifications(plaque_volume,1)
#LRNC_volume = only_calcifications(plaque_volume,2)
#IPH_volume = only_calcifications(plaque_volume,3)
#labels_out,largest_calcification,mean_size_calcification,N_of_calc,idx = number_of_calcifications(IPH_volume)
#labels_out,largest_calcification,mean_size_calcification,N_of_calc,idx = number_of_calcifications(LRNC_volume)
#labels_out,largest_calcification,mean_size_calcification,N_of_calc,idx,TotalCalcVolume = number_of_calcifications(calc_volume)
#Elongation,Flatness,Sphericity,SurfaceArea,SurfaceVolumeRatio,LargeSurfaceArea,LargeSurfaceVolumeRatio = Calc_Area_calculations(calc_volume)
#print(Elongation)
#print(Flatness)
#print(Sphericity)
#print(largest_calcification,mean_size_calcification,N_of_calc)
#print(max_stenosis_grade(slice))
#"""
#"""
#for i in test:
    #   print(i)
#    slice[i[0],i[1],i[2]] = 7
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(plaque_volume)
#    viewer.add_image(labels_out)
#    viewer.add_image(slice)
    #viewer.add_image(list)
napari.run()
#"""