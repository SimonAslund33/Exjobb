import numpy as np
import napari
import nrrd
#import os
import cv2
import glob
import json
#import pyvista as pv
from skspatial.objects import Line, Points
#from skspatial.plotting import plot_3d
from matplotlib import pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.interpolate import Rbf
#from scipy.interpolate import CubicSpline
#from mpl_toolkits import mplot3d
import math
#"E:\BiKE_0830"
def create_file_paths(Patient):
    Patient = r"/" + Patient
    Dir = r"E:/"
    suffix_comp = r"\*\RightCarotid\Batch*__composition.multi.nrrd"
    suffix_Lumen = r"\*\RightCarotid\*lumenSegmentation.nrrd"
    suffix_Wall = r"\*\RightCarotid\*wallSegmentation.nrrd"
    #suffix_Donut = r"\*\RightCarotid\CommonCarotidArtery/donut"
    suffix_save = r"\*\RightCarotid/"
    suffix_Json = r"\*\RightCarotid\Batch*__lesionReadings.json"

#filepath1 = glob.glob(r'C:\Users\simon\OneDrive\Desktop\ExJobbPlaqueInfo\BiKE_0830\*\RightCarotid\Batch*__composition.multi.nrrd')
    filepath1 = glob.glob(Dir + Patient + suffix_comp)
#print(filepath1[0])

    filepath2 = glob.glob(Dir + Patient + suffix_Lumen)
#print(filepath2[0])

    filepath3 = glob.glob(Dir + Patient + suffix_Wall)
#print(filepath3[0])
    filepath4 = glob.glob(Dir + Patient + suffix_save)
#print(filepath4)
    filepath5 = glob.glob(Dir + Patient + suffix_Json)
    return filepath1,filepath2,filepath3,filepath4,filepath5

def create_volume(f1,f2,f3,f4,f5):


#print(filepath5)
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

    #distal_list = []
    #Center_list = []
    #distal_list2 = []
    test_common = []
    test_internal = []
    C_firstz = root['cross_sections'][0]['position'][2]
    C_firsty =  root['cross_sections'][0]['position'][1]
    C_firstx =  root['cross_sections'][0]['position'][0]
    borders = [readings_d['lesions'][0]['borders'][0]['position'], readings_d['lesions'][1]['borders'][1]['position']]
    
    Lower_end = borders[0][2]+(-1)*C_firstz
    Hihger_end = borders[1][2]+(-1)*C_firstz
    #root = readings_d['initialization_points']
    #distal = root['distal_segments']  

    for section in root['cross_sections']:
        a = section['position'][2]
        a2 = section['position'][1]
        a3 = section['position'][0]
        a4 = [a3,a2,a]
        test_common.append(a4)

    for section in distal[1]['cross_sections']:
        c = section['position'][2]
        c2 = section['position'][1]
        c3 = section['position'][0]
        c4 = [c3,c2,c]
        test_internal.append(c4)
   
    
    #print(distal_list2)
    firstzC = root['cross_sections'][0]['position'][2]
    lastzC = root['cross_sections'][-1]['position'][2]
    firstyC = root['cross_sections'][0]['position'][1]
    lastyC = root['cross_sections'][-1]['position'][1]
    firstxC = root['cross_sections'][0]['position'][0]
    lastxC = root['cross_sections'][-1]['position'][0]

    firstzI = distal[1]['cross_sections'][0]['position'][2]
    lastzI = distal[1]['cross_sections'][-1]['position'][2]
    firstyI = distal[1]['cross_sections'][0]['position'][1]
    lastyI = distal[1]['cross_sections'][-1]['position'][1]
    firstxI = distal[1]['cross_sections'][0]['position'][0]
    lastxI = distal[1]['cross_sections'][-1]['position'][0]
    #lastx = distal[0]['cross_sections'][-1]['position'][0]
    test_tot = test_common + test_internal
   
    for i in test_tot:
        i[0] = i[0]-firstxC
        i[1] = i[1]-firstyC
        i[2] = i[2]-firstzC

    
    #print(test_internal)
    lengthZ = abs(firstzC - lastzI)
    #print(abs(firstxC-lastzI)/hej)
    lengthY = abs(firstyC-lastyI)
    lengthX = abs(firstxC-lastxI)

    #haha = (labelVolume.shape[2]-1)/(abs(firstzC-lastzI))
    #print(haha)
    #hehe = round(abs(firstzC-lastzC)*haha)
    #print(hehe)
    #hoho = round(abs(firstzC-firstzI)*haha)
    #print(hej,hej2,hej3)
    #print(x,y,x2,y2)
    #print(hoho)
    #middle_center = Find_center(labelVolume[:,:,hoho])
    #print(middle_center)
    #converterz = (labelVolume.shape[2]-1)/hej
    converterz = ((labelVolume.shape[2]-1)/lengthZ)
    convertery = (abs(y-y2)/lengthY)
    converterx = (abs(x-x2)/lengthX)
    #print(converterx)
    #print(convertery)
    #print(converterz)

    #hej4 = abs(firstzI-lastzI)
    #hej5 = abs(firstyI-lastyI)
    #hej6 = abs(firstxI-lastxI)

    #converterz2 = ((labelVolume.shape[2]-hehe-1)/hej4)
    #convertery2 = (abs(middle_center[1]-y2)/hej5)
    #converterx2 = (abs(middle_center[0]-x2)/hej6)
    
    Hihger_end = round(Hihger_end*converterz)
    Lower_end = round(Lower_end*converterz)
    test = [[i[0]*converterx+x,i[1]*convertery+y,i[2]*converterz] for i in test_tot]
    test = [[round(i[0]),round(i[1]),round(i[2])] for i in test]
    #print(test)
    #xt = []
    #test2 = [[i[0]*converterx2+middle_center[0],i[1]*convertery2+middle_center[1],i[2]*converterz2+hoho-2] for i in test_internal]
    #test2 = [[round(i[0]),round(i[1]),round(i[2])] for i in test2]
    #yt = []
    #zt = []
    #print(test2)
    
    
   

    return Lower_end,Hihger_end,test






def Find_center(image):
    centroid = np.mean(np.argwhere(image==6),axis=0)
    x = round(centroid[0])
    y = round(centroid[1])
    return [x,y]

def unwrap_slice(slice):
    slice = np.array(slice)
    #print(slice.shape)
    unwrapped_first_half = []
    unwrapped_second_half = []
    slice = np.pad(slice,((30,30),(30,30)),mode ='constant', constant_values=0)
    center = np.where(slice == 7)
    x1 = center[0][0]
    y1 = center[1][0]
    #slice = np.pad(slice,((30,30),(30,30)),mode ='constant', constant_values=0)
    #print(slice.shape)
    #print(x1)
    #print(y1)
    slice = slice[x1-31:x1+31, y1-31:y1+31]
    #print(slice.shape)
    center = np.where(slice == 7)
    x1 = center[0][0]
    y1 = center[1][0]

    #slice.shape[1]
    Radius = 1
    angle = 0
    while angle < 180:
        #print(angle)
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
        #print(this_angle)
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
        #print(len(a))
        total_unwrap2.append(a)

    total_unwrap = np.array(total_unwrap2)
    return total_unwrap

def vertical_slices(Center_line, labelVolume, H, L):
    #for i in Center_line:
    #    labelVolume[i[0],i[1],i[2]] = 7

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
                
                        
    return slice, volume_short, labelVolume

def new_center(image, CL):
    #centers = []
    for j in range(image.shape[2]):
        thresh = cv2.inRange(image[:,:,j], 6, 6)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        #CL = np.where(image[:,:,j] == 7)
        #print(contours)
        distance = 1000
        for i in contours:
            point = (sum(i)/len(i))
            #print(point)
            point = [round(k) for k in point[0]]
            dist = np.sqrt( (point[1] - CL[j][0])**2 + (point[0] - CL[j][1])**2 )
            #print(dist)
            if dist < distance:
                distance = dist
                closest_point = point
            
        image[closest_point[1],closest_point[0],j] = 7
        
    return image

def main():
    f1,f2,f3,f4,f5 = create_file_paths("BiKE_0846")
    labelVolume, reading, LumenData = create_volume(f1,f2,f3,f4,f5)
    cent = Find_center(labelVolume[:,:,0])
    end_center = Find_center(labelVolume[:,:,labelVolume.shape[2]-1])
    #middle_center = Find_center(labelVolume[:,:,115])
    #print(middle_center)
    print(end_center)
    print(cent)
    L,H,test = build_Center_Line(cent[0],cent[1],end_center[0],end_center[1],reading,labelVolume)

    #Total = C + d2 + d
    
    slice, label2, labelVolume2 = vertical_slices(test, labelVolume,H,L)
    
    #label2 = LumenData[:,:,L:H]
    #print(slice.shape)
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
    #slice = np.load(f4[0] + r"Slices.txt.npy")
#print(slice.shape)

    
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
    for i in test:
    #   print(i)
       labelVolume[i[0],i[1],i[2]] = 7
    with napari.gui_qt():
        viewer = napari.Viewer()
    #viewer.add_image(total_unwrap)
        viewer.add_image(label2)
        viewer.add_image(slice)
        viewer.add_image(labelVolume)
        viewer.add_image(unwraps)
    napari.run()

    
    
main()