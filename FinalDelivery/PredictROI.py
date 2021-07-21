import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import io
from glob import glob
from os import listdir
import torch
import sys
import os
from datetime import datetime





## Enter the image path here
img_path=r"E:\MBDS Materials\BMDSIS\\Final Delivery\Testing Folder\Testing Image"

## Enter the destination path here
destination_path=r"E:\MBDS Materials\BMDSIS\\Final Delivery\Testing Folder\ROI_Predict"










def MarkObject(img):
    imax=img.shape[0]
    jmax=img.shape[1]
    objectness=np.zeros((imax,jmax))
    direction=[[0,1],[1,0],[0,-1],[-1,0]]

    def dfs_markobject(img,i,j):
        if i<0 or j<0 or i>=imax or j>=jmax or (img[i][j]==0).all() or objectness[i][j]==1:
            return

        objectness[i][j]=1

        for d in direction:
            dfs_markobject(img,i+d[0],j+d[0])
        return
    
    for i in range(0,imax):
        for j in range(0,jmax):
            if objectness[i][j]!=1 and (img[i][j]!=0).any():
                dfs_markobject(img,i,j)

            else:
                continue
    
    return objectness

def dfs(img,coordinates,i,j,visited):
    if i<0 or i>=img.shape[0] or j<0 or j>=img.shape[1] or img[i][j]==0 or visited[i][j]==1:
        return
        
    visited[i][j]=1;
    if i<coordinates[0]:
        coordinates[0]=i;
    if i>coordinates[1]:
        coordinates[1]=i;
    if j<coordinates[2]:
        coordinates[2]=j;
    if j>coordinates[3]:
        coordinates[3]=j;
    
    dfs(img,coordinates,i+1,j,visited);
    dfs(img,coordinates,i,j+1,visited);
    dfs(img,coordinates,i-1,j,visited);
    dfs(img,coordinates,i,j-1,visited);

def GetCoordinates(img_ori,img,istart,iend,jstart,jend):
    i_diff_start=1
    i_diff_end=1
    j_diff_start=1
    j_diff_end=1

    if istart==0:
        i_diff_start=0
    if iend==1080:
        i_diff_end=0
    if jstart==0:
        j_diff_start=0
    if jend==1280:
        j_diff_end=0
    visited=np.zeros((iend-istart+i_diff_start+i_diff_end,jend-jstart+j_diff_start+j_diff_end))
    img_temp=img[istart-i_diff_start:iend+i_diff_end,jstart-j_diff_start:jend+j_diff_end]
    #print(img_temp.shape)
    #print(visited.shape)
    clist=[]
    for i in range(0,50+i_diff_end+i_diff_start):
        for j in range(0,50+j_diff_end+j_diff_start):
            #print("Done with {},{}".format(i,j))
            #print(img_temp.shape)
            if img_temp[i][j]==0 or visited[i][j]==1:
                continue
            c=[i,i,j,j]
            dfs(img_temp,c,i,j,visited)
            #print("Top:{} Bottom:{} Left:{} Right:{}".format(c[0],c[1],c[2],c[3]))
            c[0]=c[0]+istart-i_diff_start
            c[1]=c[1]+istart-i_diff_start
            c[2]=c[2]+jstart-j_diff_start
            c[3]=c[3]+jstart-j_diff_start
            
            ## If top = bottom, left = right, and is inside our predefined boundary
            if c[0]!=c[1] and c[2]!=c[3] and c[0]>=istart and c[1]<iend and c[2]>=jstart and c[3]<jend:
                #print("To Add: Top:{} Bottom:{} Left:{} Right:{}".format(c[0],c[1],c[2],c[3]))
                if not (img_ori[c[0]:c[1]+1,c[2]:c[3]+1,0]==0).all() and  not (img_ori[c[0]:c[1]+1,c[2]:c[3]+1,1]==0).all() and not (img_ori[c[0]:c[1]+1,c[2]:c[3]+1,0]==0).all():
                    clist.append(c)
    
    if len(clist)>0:
        clist=np.array(clist)
        df=pd.DataFrame({'top':clist[:,0],'bottom':clist[:,1],'left':clist[:,2],'right':clist[:,3]})
        return df
    else:
        return None

def Transform(img):
    test=np.swapaxes(img,0,2)
    test=np.swapaxes(test,1,2)
    return torch.tensor(np.array([test],dtype='f'))

def GetObject(img,top,bottom,left,right):
    roi=img[top:bottom+1,left:right+1]
    return cv2.resize(roi, dsize=(17, 17), interpolation=cv2.INTER_CUBIC)

## Prediction method for single model
def Predict(Model,X):
    test_pred = Model(X)
    res =test_pred.data
    y_pred=torch.max(res,1)[1]
    return y_pred

## Prediction method for ensembled method
def EsemblePredict(X):
    all_res=[]
    for x in X:
        x_in=torch.tensor(np.reshape(x,(1,3,17,17)))
        res=[Predict(model_6_2_2,x_in)[0].data.numpy(),Predict(model_6_2_3,x_in)[0].data.numpy(),Predict(model_6_3_3,x_in)[0].data.numpy()]
        if res.count(0)>res.count(1):
            all_res.append(0)
        else:
            all_res.append(1)
    return np.array(all_res)

def Scale(top,bottom,left,right):
    if top-2>=0:
        top=top-2
    elif top-1>=0:
        top=top-1
    
    if left-2>=0:
        left=left-2
    elif left-1>=0:
        left=left-1

    if bottom+3<1080:
        bottom=bottom+3
    elif bottom+2<1080:
        bottom=bottom+2
    elif bottom+1<1080:
        bottom=bottom+1
    
    if right+3<1280:
        right=right+3
    elif right+2<1280:
        right=right+2
    elif right+1<1280:
        right=right+1
    
    return top,bottom,left,right

class Model_VGG(torch.nn.Module):
    def __init__(self):
        super(Model_VGG, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=1),
            #BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            #BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            #BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 512, kernel_size=2, stride=1, padding=1),
            #BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(128,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Model_VGG_v2(torch.nn.Module):
    def __init__(self):
        super(Model_VGG_v2, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 512, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(128,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

if __name__=='__main__':

    sys.setrecursionlimit(19999999)

    ## Load models
    model_6_2_1=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_2_1.pkl')
    model_6_2_2=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_2_2.pkl')
    model_6_2_3=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_2_3.pkl')
    model_6_3_1=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_3_1.pkl')
    model_6_3_2=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_3_2.pkl')
    model_6_3_3=torch.load('E:\MBDS Materials\BMDSIS\Final Delivery\model_6_3_3.pkl')

    image_files=listdir(img_path)

    for img_file in image_files:

        start = datetime.now()

        current_time = start.strftime("%H:%M:%S")
        print("Start to process image {}".format(img_file))
        print("Start Time = {}".format(current_time))

        ## Read source image
        img=io.imread("{}\{}".format(img_path,img_file)) 

        ## Mark object area in image
        img_obj=MarkObject(img)

        ## Scan the image to extract object coordinates
        obj_df=pd.DataFrame(columns=['top','bottom','left','right'])

            
        for i in range(0,1031,2):
            print("Processing image {} now: row {} to row {}".format(img_file,i,i+50))
            for j in range(0,1231,2):
                if (img_obj[i:i+50,j:j+50]==0).sum()<=50 or (img_obj[i:i+50,j:j+50]==1).sum()<=30:
                    continue
                cur_co_df=GetCoordinates(img,img_obj,i,i+50,j,j+50)
                if cur_co_df is not None:
                    obj_df=obj_df.append(cur_co_df).drop_duplicates().reset_index(drop=True)


        

        ## Classify the object
        label=0
        group_label=0

        temp_destination_path="{}\{} - Predicted ROI".format(destination_path,img_file[:-4])

        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        if not os.path.exists(temp_destination_path):
            os.mkdir(temp_destination_path)

        for index,row in obj_df.iterrows():
            top=row['top']
            bottom=row['bottom']
            left=row['left']
            right=row['right']

            obj=GetObject(img,top,bottom,left,right)
            obj_tensor=Transform(obj)
            #print(model(obj_tensor)[0])
            if EsemblePredict(obj_tensor)[0]==1:
                top,bottom,left,right=Scale(top,bottom,left,right)
                coordinates=pd.DataFrame({"X":[left,left,right,right],'Y':[bottom,top,top,bottom]})

                final_path=temp_destination_path+"\\XY_Coordinates_0{}.csv".format(label)
                    
                coordinates.to_csv(final_path)
                label+=1
        
        print("Totally {} ROIs predicted, saved to {}".format(label,temp_destination_path))

        end = datetime.now()

        current_time = end.strftime("%H:%M:%S")
        print("End Time =", current_time)

        time_delta = (end - start)
        total_seconds = time_delta.total_seconds()
        minutes = total_seconds/60

        print("Whole process has taken {} mins for image {}.".format(minutes,img_file))