import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import io
import torch
from glob import glob
import sys

def Transform(img):
    test=np.swapaxes(img,0,2)
    test=np.swapaxes(test,1,2)
    return torch.tensor([np.array(test,dtype='f')]).float()

def GetObject(img,top,bottom,left,right):
    roi=img[top:bottom+1,left:right+1]
    return cv2.resize(roi, dsize=(17, 17), interpolation=cv2.INTER_CUBIC)

def Predict(Model,X):
    test_pred = Model(X)
    res =test_pred.data
    y_pred=torch.max(res,1)[1]
    return y_pred

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
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(64,2)
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
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(64,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

import time
import signal


def test(i):
    time.sleep(i % 4)
    print("{} within time".format(i))
    return i

if __name__=='__main__':
    folders_composite=glob("E:/MBDS Materials/BMDSIS/Data/Composite Images/*/")
    labels=pd.read_csv('E:/MBDS Materials/BMDSIS/Data/labels.csv',index_col=0)
    
    test_label=list(labels.iloc[32:38]['label'])
    
    sys.setrecursionlimit(1999999999)


    img_file_path="E:\\MBDS Materials\\BMDSIS\\Data\\Composite Images\\{}-Data Preparation\\composite_{}.tif".format(test_label[1],test_label[1])
    img=io.imread(img_file_path)
    print(test_label[1])

    co_df=pd.DataFrame(columns=['top','bottom','left','right'])




    for i in range(0,1031,5):
        print("Processing row {} to row {}".format(i,i+50))
        for j in range(0,1231,5):
            if (img_obj[i:i+50,j:j+50]==0).sum()<=50:
                continue
            cur_co_df=GetCoordinates(img_obj,i,i+50,j,j+50)
            if cur_co_df is not None:
                co_df=co_df.append(cur_co_df).reset_index(drop=True)
            
    co_df=co_df.drop_duplicates().reset_index(drop=True)

    print(co_df)
    co_df.to_csv('E:/MBDS Materials/BMDSIS/Data/co_df.csv')

    #print(GetCoordinates(img_obj,435,435+50,615,615+50))
            
                
        
    
        

    