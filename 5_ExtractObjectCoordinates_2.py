import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import io
from glob import glob
from os import listdir
import sys

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

if __name__=='__main__':
    folders_composite=glob("E:\\MBDS Materials\\BMDSIS\\Data\\Composite Images/*/")
    labels=pd.read_csv('E:\\MBDS Materials\\BMDSIS\\Data\\labels.csv',index_col=0)

    sys.setrecursionlimit(19999999)

    for k in range(0,40):

        img_file_path=folders_composite[k]+"composite_{}.tif".format(labels.iloc[k].values[0])
        img=io.imread(img_file_path)
        
        img_non_golgy_file_path=folders_composite[k]+"non_golgy_object.tif"
        img_non_golgy_obj=io.imread(img_non_golgy_file_path)

        non_golgi_df=pd.DataFrame(columns=['top','bottom','left','right'])
        
        ## Extract coordinates for non-golgy objects
        for i in range(0,1031,2):
            print("Processing {} - {} now: row {} to row {}".format(k,labels.iloc[k].values[0],i,i+50))
            for j in range(0,1231,2):
                if (img_non_golgy_obj[i:i+50,j:j+50]==0).sum()<=50 or (img_non_golgy_obj[i:i+50,j:j+50]==1).sum()<=30:
                    #print('Skip!')
                    continue
                cur_co_df=GetCoordinates(img,img_non_golgy_obj,i,i+50,j,j+50)
                if cur_co_df is not None:
                    non_golgi_df=non_golgi_df.append(cur_co_df).reset_index(drop=True)

        non_golgi_df=non_golgi_df.drop_duplicates().reset_index(drop=True)
        non_golgy_obj_co_file_path=folders_composite[k]+"\\{}_non_golgy_coordinates.csv".format(labels.iloc[k].values[0])
        non_golgi_df.to_csv(non_golgy_obj_co_file_path)

        print('Done with {}'.format(labels.iloc[k].values[0]))