import numpy as np
import napari
import nrrd
import os
import cv2
import glob
import json
from numba import jit, cuda
#import pyvista as pv
from skspatial.objects import Line, Points
#from skspatial.plotting import plot_3d
from matplotlib import pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.interpolate import Rbf
#from scipy.interpolate import CubicSpline
#from mpl_toolkits import mplot3d
import pandas as pd
import math
#"E:\BiKE_0830"
#E:\wilist_BiKE_Group001_EJ\BiKE_0568\wi-F3F85BCF\LeftCarotid\BatchProcessor_20220925194821253636__lesionReadings.json
xls = pd.ExcelFile(r'C:\Users\simon\Downloads\BiKE_imported_csv.xlsx')
df1 = pd.read_excel(xls, 'BiKE Elucid plus operation')

#xls = pd.ExcelFile(r'E:\BIKE missing DM.xlsx')
#df1 = pd.read_excel(xls, 'Sheet1')
def create_file_paths(Patient):
    print(Patient)
    Patient = str(Patient)
    #list = np.unique(df1.iloc[:,1])
    #print(list)
    for i in range(len(df1)):
        #print(df1.iloc[i,1])
        if df1.iloc[i,1] == Patient:
            #print("found!")
            s = df1.iloc[i,12]
            print(s)
            break
    side = s + r"Carotid"
    Patient = r"/" + Patient + r"\*/"
    Dir = r"E:/*"
    suffix_comp = r"\*__composition.multi.nrrd"
    suffix_Lumen = r"\*lumenSegmentation.nrrd"
    suffix_Wall = r"\*wallSegmentation.nrrd"
    #suffix_Donut = r"\*\RightCarotid\CommonCarotidArtery/donut"
    suffix_save = r"/"
    suffix_Json = r"\*__lesionReadings.json"

#filepath1 = glob.glob(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0830\*\RightCarotid\Batch*__composition.multi.nrrd')
    filepath1 = glob.glob(Dir + Patient + side + suffix_comp)
#print(filepath1[0])
    
    filepath2 = glob.glob(Dir + Patient + side + suffix_Lumen)
#print(filepath2[0])

    filepath3 = glob.glob(Dir + Patient + side + suffix_Wall)
#print(filepath3[0])
    filepath4 = glob.glob(Dir + Patient + side + suffix_save)
#print(filepath4)
    filepath5 = glob.glob(Dir + Patient + side + suffix_Json)
    print(filepath5)
    return filepath1,filepath2,filepath3,filepath4,filepath5
    
def create_volume(f1,f2,f3,f4,f5):


#print(filepath5)
    print(f5)
    readings_d = json.load(open(f5[0]))
    CompositionData1, header2 = nrrd.read(f1[0])

    LumenData, header2 = nrrd.read(f2[0])
    WallData, header3 = nrrd.read(f3[0])
    print(CompositionData1.shape)

    comp = np.zeros_like(CompositionData1[:,:,:,0])

    for i in range(5):
    # 1 - CALC
    # 2 - LRNC
    # 3 - IPH
    # 4 - PVAT
    # 5 - MATX
        hej = CompositionData1[:,:,:,i]
        i = i+1

        if i == 1:
            hej[hej > 0] = 0
            hej[hej < 0] = i
        elif i == 2:
            hej[hej > 0] = 0
            hej[hej < 0] = i
        elif i == 3:
            hej[hej > 0] = 0
            hej[hej < 0] = i
        elif i == 4:
            hej[hej > 0] = 0
            hej[hej < 0] = 0
        elif i == 5:
            hej[hej > 0] = 0
            hej[hej < 0] = i
        else:
            print("what")
        comp = comp + hej
  

    LumenData[LumenData > 0] = 0
    LumenData[LumenData < 0] = 6



#6 - LUMEN
    labelVolume = comp + LumenData



    Original_shape = labelVolume.shape
    return labelVolume,readings_d,LumenData
    

def build_Center_Line(x,y,x2,y2, readings_d, labelVolume):

    root = readings_d['root_segment']

    distal = root['distal_segments']    

    
    test_common = []
    test_internal = []
    C_firstz = root['cross_sections'][0]['position'][2]
    C_firsty =  root['cross_sections'][0]['position'][1]
    C_firstx =  root['cross_sections'][0]['position'][0]
    
    lower = 0
    higher = 0
    distance = 0
    """
    for i in range(len(readings_d['lesions'][0]['borders'])):
        
        zC = readings_d['lesions'][0]['borders'][i]['position'][2]
        
        for j in range(len(readings_d['lesions'][1]['borders'])):
            
            zI = readings_d['lesions'][1]['borders'][j]['position'][2]
            
            if abs(abs(zC) - abs(zI)) > distance:
                lower = zC
                higher = zI
                distance = dist = abs(abs(zC) - abs(zI))
    #print(higher)
    #print(lower) 
    borders = [readings_d['lesions'][0]['borders'][0]['position'], readings_d['lesions'][1]['borders'][0]['position']]
    #print(borders)
    
    Lower_end = lower+(-1)*C_firstz
    Hihger_end = higher+(-1)*C_firstz
    #print(Lower_end)
    #print(Hihger_end)
    """
    borders = [readings_d['lesions'][0]['borders'][0]['position'], readings_d['lesions'][0]['borders'][1]['position']]
    Lower_end = borders[0][2]+(-1)*C_firstz
    Hihger_end = borders[1][2]+(-1)*C_firstz

    for section in root['cross_sections']:
        a = section['position'][2]
        a2 = section['position'][1]
        a3 = section['position'][0]
        a4 = [a3,a2,a]
        test_common.append(a4)
    
    if readings_d['root_segment']['distal_segments'][0]["segment_name"] == "InternalCarotidArtery":
        path = distal[0]['cross_sections']
    #if readings_d['root_segment']['distal_segments'][1]["segment_name"] == "InternalCarotidArtery":
    #    path = distal[1]['cross_sections']
    for section in path:
        c = section['position'][2]
        c2 = section['position'][1]
        c3 = section['position'][0]
        c4 = [c3,c2,c]
        test_internal.append(c4)
   
    
    
    firstzC = root['cross_sections'][0]['position'][2]
    lastzC = root['cross_sections'][-1]['position'][2]
    firstyC = root['cross_sections'][0]['position'][1]
    lastyC = root['cross_sections'][-1]['position'][1]
    firstxC = root['cross_sections'][0]['position'][0]
    lastxC = root['cross_sections'][-1]['position'][0]
    
    firstzI = path[0]['position'][2]
    
    lastzI = path[-1]['position'][2]
    
    firstyI = path[0]['position'][1]
    lastyI = path[-1]['position'][1]
    firstxI = path[0]['position'][0]
    lastxI = path[-1]['position'][0]
    
    test_tot = test_common + test_internal
   
    for i in test_tot:
        i[0] = i[0]-firstxC
        i[1] = i[1]-firstyC
        i[2] = i[2]-firstzC

    
    
    lengthZ = abs(firstzC - lastzI)
    lengthY = abs(firstyC-lastyI)
    lengthX = abs(firstxC-lastxI)

   
    converterz = ((labelVolume.shape[2]-1)/lengthZ)
    convertery = (abs(y-y2)/lengthY)
    converterx = (abs(x-x2)/lengthX)
    
    
    Hihger_end = round(Hihger_end*converterz)
    Lower_end = round(Lower_end*converterz)
    test = [[i[0]*converterx+x,i[1]*convertery+y,i[2]*converterz] for i in test_tot]
    test = [[round(i[0]),round(i[1]),round(i[2])] for i in test]
    #print(test)
    
    
   

    return Lower_end,Hihger_end,test






def Find_center(image):
    centroid = np.mean(np.argwhere(image==6),axis=0)
    x = round(centroid[0])
    y = round(centroid[1])
    return [x,y]
@jit(target_backend='cuda')
def unwrap_slice(slice):
    slice = np.array(slice)
    
    unwrapped_first_half = []
    unwrapped_second_half = []
    slice = np.pad(slice,((30,30),(30,30)),mode ='constant', constant_values=0)
    center = np.where(slice == 7)
    x1 = center[0][0]
    y1 = center[1][0]
    
    slice = slice[x1-31:x1+31, y1-31:y1+31]
    
    center = np.where(slice == 7)
    x1 = center[0][0]
    y1 = center[1][0]

    
    Radius = 1
    angle = 0
    while angle < 180:
        
        radians = angle*2*math.pi/360
        
        x2 = math.cos(radians)*Radius+x1
        y2 = math.sin(radians)*Radius+y1
  
        p1 = np.array([x1,y1])
        p2 = np.array([x2,y2])
        this_angle=[]
        if (45 >= angle >= 0) or (179 >= angle >= 135):
            
            angle += 1
            for i in range(slice.shape[0]):
                distance = 1000
                for j in range(slice.shape[1]):
                    p3 = np.array([i,j])
                    d = abs(np.cross(p2-p1, p1-p3)/np.linalg.norm(p2-p1))
                    
                    if (d < distance):
                        distance = d
                        save = [i,j]
                        
                
                this_angle.append(slice[save[0],save[1]])         
        else:
            
            angle += 1
            for i in range(slice.shape[1]):
                distance = 1000
                for j in range(slice.shape[0]):
                    p3 = np.array([j,i])
                    d = abs(np.cross(p2-p1, p1-p3)/np.linalg.norm(p2-p1))
                    
                    if (d < distance):
                        distance = d
                        save = [j,i]
                        
                this_angle.append(slice[save[0],save[1]])
        length = len(this_angle) 
        
        end = length-1
       
        half = this_angle.index(7.0)
        
        unwrapped_first_half.append(this_angle[0:half+1])
        unwrapped_second_half.append(this_angle[half:end])         
            
       
    return unwrapped_first_half,unwrapped_second_half            
                
                    
def scale_unwrap(first_half, second_half):

    [i.reverse() for i in second_half]

    total_unwrap = first_half[0:90] + first_half[90:135] + second_half[135:180] + second_half[0:45] + second_half[45:90] + second_half[90:135] + first_half[135:180]

    total_unwrap2 = []
    for i in total_unwrap:
        length = len(i)
        a = i[-30:length]
        
        total_unwrap2.append(a)

    total_unwrap = np.array(total_unwrap2)
    return total_unwrap

@jit(target_backend='cuda')
def vertical_slices(Center_line, labelVolume, H, L):
    

    volume_short = labelVolume[:,:,L:H]
    volume_short = np.array(volume_short)
    
    arr = np.array(Center_line)
    
    new = []
    for i in range(len(arr)):
    
        if H >= arr[i][2] and arr[i][2] >= L:
            
            new.append(list(arr[i]))

    
    new_end = len(new)
   
    points = Points(
        new[0:new_end-1]
    )
   
    slice = np.zeros([labelVolume.shape[0], labelVolume.shape[1], len(range(2,new_end-2))])
    
    
    for k in range(2,new_end-2):
    
        print(k)
        
        line_fit = Line.best_fit(points[(k-2):(k+2)])
        
        point_in_plane = points[k]
        
        for i in range(volume_short.shape[0]):
            for j in range(volume_short.shape[1]):
                distance = 1000
                for h in range(volume_short.shape[2]):
                    point=[i,j,h+points[0][2]]
                    dist = abs(np.dot((point-point_in_plane), line_fit.direction ))
                    
                    if dist < distance:
                        distance = dist
                        
                        slice[i,j,k-2] = volume_short[i,j,h]
                
                        
    return slice, volume_short, labelVolume

def new_center(image, CL):
    
    print(image.shape)
    print(len(CL))
    for j in range(image.shape[2]):
        thresh = cv2.inRange(image[:,:,j], 6, 6)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        distance = 1000
        for i in contours:
            point = (sum(i)/len(i))
            
            point = [round(k) for k in point[0]]
            dist = np.sqrt( (point[1] - CL[j][0])**2 + (point[0] - CL[j][1])**2 )
            
            if dist < distance:
                distance = dist
                closest_point = point
            
        image[closest_point[1],closest_point[0],j] = 7
        
    return image

def main():
    #for filename in os.listdir(r"E:\Controlled_Patients"):
    #List = ["BiKE_0717","BiKE_0735","BiKE_0739","BiKE_0784","BiKE_0881","BiKE_0890","BiKE_0906",
     #       "BiKE_0908","BiKE_0918","BiKE_0944"
#]  #   
    #for i in List
    f1,f2,f3,f4,f5 = create_file_paths("ANONNF13LG1FK")
    labelVolume, reading, LumenData = create_volume(f1,f2,f3,f4,f5)
    cent = Find_center(labelVolume[:,:,0])
    end_center = Find_center(labelVolume[:,:,labelVolume.shape[2]-1])
    
    L,H,test = build_Center_Line(cent[0],cent[1],end_center[0],end_center[1],reading,labelVolume)
    print(L,H)
    label2 = labelVolume[:,:,L:H]
        
    
    slice, label2, labelVolume2 = vertical_slices(test, labelVolume,H,L)
        
    
    
    np.save(f4[0] + r"Slices.txt.npy", slice)
    
    np.save(f4[0] + r"Plaque_volume.txt", label2)
    
    slice = np.load(f4[0] + r"Slices.txt.npy")
    Unwrap_list = np.zeros([360,30, slice.shape[2]])
    slice = new_center(slice,test)
    for i in range(slice.shape[2]):
        print(i)
        first_half,second_half = unwrap_slice(slice[:,:,i])
        total_unwrap = scale_unwrap(first_half, second_half)
        Unwrap_list[:,:,i] = total_unwrap

    np.save(f4[0] +r"Unwraps.txt.npy", Unwrap_list)
    unwraps = np.load(f4[0] +r"Unwraps.txt.npy")
    
    
    
    
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

# plotting the points
    pts = test
    for p in pts:
        ax.scatter(p[0], p[1], p[2], zdir='z', c='r')

# plotting lines for each point pair
    

    ax.legend()
    ax.set_xlim3d(0, 357)
    ax.set_ylim3d(0, 357)
    ax.set_zlim3d(0, 357)

    plt.show()
    """
    """
    for i in test:
    #   print(i)
       labelVolume[i[0],i[1],i[2]] = 7
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(labelVolume)
        #viewer.add_image(plaque_volume)
        viewer.add_image(label2)
        #viewer.add_image(slice)
        #viewer.add_image(unwraps)
    napari.run()
    """
    
    
#main()