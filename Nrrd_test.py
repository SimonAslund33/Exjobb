import numpy as np
import napari
import nrrd
import os
import cv2
import glob
import json
import pyvista as pv
from skspatial.objects import Line, Points
from skspatial.plotting import plot_3d
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
from scipy.interpolate import CubicSpline
from mpl_toolkits import mplot3d
import math

Patient = r"\BiKE_0846"
Dir = r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo"
suffix_comp = r"\*\RightCarotid\Batch*__composition.multi.nrrd"
suffix_Lumen = r"\*\RightCarotid\*lumenSegmentation.nrrd"
suffix_Wall = r"\*\RightCarotid\*wallSegmentation.nrrd"
suffix_Donut = r"\*\RightCarotid\CommonCarotidArtery/donut"
readings_d = json.load(open(r"C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0846\wi-266BCEEA\RightCarotid\BatchProcessor_20221031181135482225__lesionReadings.json"))

#filepath1 = glob.glob(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0830\*\RightCarotid\Batch*__composition.multi.nrrd')
filepath1 = glob.glob(Dir + Patient + suffix_comp)
#print(filepath1[0])

filepath2 = glob.glob(Dir + Patient + suffix_Lumen)
#print(filepath2[0])

filepath3 = glob.glob(Dir + Patient + suffix_Wall)
#print(filepath3[0])


CompositionData1, header2 = nrrd.read(filepath1[0])

LumenData, header2 = nrrd.read(filepath2[0])
WallData, header3 = nrrd.read(filepath3[0])


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

    

def build_Center_Line(x,y):

    root = readings_d['root_segment']

    distal = root['distal_segments']    

    distal_list = []
    Center_list = []
    distal_list2 = []
    C_firstz = root['cross_sections'][0]['position'][2]
    C_firsty =  root['cross_sections'][0]['position'][1]
    C_firstx =  root['cross_sections'][0]['position'][0]
    borders = [readings_d['lesions'][0]['borders'][0]['position'], readings_d['lesions'][0]['borders'][2]['position']]
    
    Lower_end = borders[0][2]+(-1)*C_firstz
    Hihger_end = borders[1][2]+(-1)*C_firstz
    

    
    for section in root['cross_sections']:
        a = section['position'][2]+(-1)*C_firstz
        a2 = section['position'][1]+(-1)*C_firsty
        a3 = section['position'][0]+(-1)*C_firstx
        a4 = [a3,a2,a]
        Center_list.append(a4)
    
    for section in distal[0]['cross_sections']:
       
        b = section['position'][2]+(-1)*C_firstz
        b2 = section['position'][1]+(-1)*C_firsty
        b3 = section['position'][0]+(-1)*C_firstx
        b4 = [b3,b2,b]
        distal_list.append(b4)

    for section in distal[1]['cross_sections']:
        c = section['position'][2]+(-1)*C_firstz
        c2 = section['position'][1]+(-1)*C_firsty
        c3 = section['position'][0]+(-1)*C_firstx
        c4 = [c3,c2,c]
        distal_list2.append(c4)
    firstz = root['cross_sections'][0]['position'][2]
    lastz = distal[1]['cross_sections'][-1]['position'][2]
    hej = (-1)*firstz - (-1)*lastz
    
    converterz = (labelVolume.shape[2]-1)/hej
    Hihger_end = round(Hihger_end*converterz)
    Lower_end = round(Lower_end*converterz)
    
    
    Center_list = [[round(j*converterz) for j in i] for i in Center_list]
    Center_list = [[i[0]+x,i[1]+y,i[2]] for i in Center_list]
    distal_list = [[round(j*converterz) for j in i] for i in distal_list]
    distal_list = [[i[0]+x,i[1]+y,i[2]] for i in distal_list]
    distal_list2 = [[round(j*converterz) for j in i] for i in distal_list2]
    distal_list2 = [[i[0]+x,i[1]+y,i[2]] for i in distal_list2]

    return Center_list ,distal_list,distal_list2,Lower_end,Hihger_end






def Find_center(image):
    centroid = np.mean(np.argwhere(image),axis=0)
    x = round(centroid[0])
    y = round(centroid[1])
    return [x,y]

def unwrap_slice(slice):
    slice = np.array(slice)
    unwrapped_first_half = []
    unwrapped_second_half = []
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

def vertical_slices(Center_line, labelVolume, H, L):
    for i in Center_line:
        labelVolume[i[0],i[1],i[2]] = 7

    volume_short = labelVolume[:,:,L:H]
    volume_short = np.array(volume_short)
    #print(volume_short.shape)
    arr = np.array(Center_line)
    #print(L,H)
    #print(arr)
    new = []
    for i in range(len(arr)):
    
        if H >= arr[i][2] and arr[i][2] >= L:
            #arr[i][2] = arr[i][2]-84
            new.append(list(arr[i]))

    #new = np.array(new)
    new_end = len(new)
    #print(new)
    #print(new_end)
    points = Points(
        new[0:new_end-1]
    )
    #print(points)
    #print(new)

    #d = line_fit.direction[0] * points[0,0] + line_fit.direction[1] * points[0,1] + line_fit.direction[2] * points[0,2]
    #slice = []
    slice = np.zeros([labelVolume.shape[0], labelVolume.shape[1], len(range(2,new_end-2))])
    #print(slice.shape)
    
    for k in range(2,new_end-2):
    #for k in range(2,4):
        print(k)
        #haha = len(range(48,new_end-2))
        #print(haha)
        line_fit = Line.best_fit(points[(k-2):(k+2)])
        #print(line_fit)
        #point_in_plane=line_fit.point
        point_in_plane = points[k]
        #print(point_in_plane)
        for i in range(volume_short.shape[0]):
            for j in range(volume_short.shape[1]):
                distance = 1000
                for h in range(volume_short.shape[2]):
                    point=[i,j,h+points[0][2]]
                    dist = abs(np.dot((point-point_in_plane), line_fit.direction ))
                    #print(dist)
                    if dist < distance:
                        distance = dist
                        #print(h)
                        slice[i,j,k-2] = volume_short[i,j,h]
                
                        
    return slice, volume_short


cent = Find_center(labelVolume[:,:,0])
C,d,d2,L,H = build_Center_Line(cent[0],cent[1])

Total = C + d2
slice, label2 = vertical_slices(Total, labelVolume,H,L)

print(slice.shape)
Unwrap_list = np.zeros([360,30, slice.shape[2]])
for i in range(slice.shape[2]):
    first_half,second_half = unwrap_slice(slice[:,:,i])
    total_unwrap = scale_unwrap(first_half, second_half)
    Unwrap_list[:,:,i] = total_unwrap





with napari.gui_qt():
    viewer = napari.Viewer()
    #viewer.add_image(total_unwrap)
    viewer.add_image(slice)
    #viewer.add_image(labelVolume)
    viewer.add_image(label2)
    viewer.add_image(Unwrap_list)
napari.run()