# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:21:52 2019

@author: sudhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance

data = pd.read_csv('D:\Hackathon\output.csv')

from sklearn.cluster import KMeans

clusters = 8
MYSEED = 9

estimator = KMeans(n_clusters = clusters,n_init = 25, max_iter = 600, random_state=MYSEED)

estimator.fit(data.loc[:, ['lgt', 'lat']])
plt.scatter(data['lgt'], data['lat'], c=estimator.labels_)
plt.show()
plt.close()

df = pd.DataFrame(data)
df['group'] = estimator.labels_
##data.to_csv("D:/Hackathon/data_Sudhu1.csv", encoding="utf-8")
#
l = dict()
for i,row in df.iterrows():
    team = row['ID']
    group  = row['group']
    #print(team, group)
    if group in l:
        l[group].append(team)
    else:
        l[group] = [team]

l2 = [l[g] for g in l]
print(l2)


#print(estimator.labels_)

#print(estimator.cluster_centers_)

clust_centre = estimator.cluster_centers_


sort = np.zeros((max(estimator.labels_)+1, len(data['lgt'])))
maxcol = 0
clust_size = []
for idx in range(0,max(estimator.labels_)+1):
    chk = data.ID[estimator.labels_ == idx]
    chk = np.array(chk)
    clust_size.append(len(chk))
    
    if(len(chk) > maxcol):
        maxcol = len(chk)
    for idxcol in range(0,len(chk)):
        sort[idx][idxcol] =  chk[idxcol]  
        
maxcol = maxcol + 10        
sort = np.delete(sort,np.s_[maxcol :len(data['lgt'])],axis=1)

clust_size = np.array(clust_size)

after_sort = np.zeros((max(estimator.labels_)+1,maxcol))


sort_clust_centre = []

for i in range(0,max(estimator.labels_)+1):
    idx = np.where(clust_size == max(clust_size))
    idx_row = idx[0][0]
    sort_clust_centre.append(clust_centre[idx_row])
    for j in range(0,maxcol):
        after_sort[i][j] = sort[idx_row][j]
        
        clust_size[idx_row] = 0
        

#print(after_sort)


  
#from scipy.spatial import distance

def rearrange(arr):
    for i in range(0,len(arr)-1):
        stop = False
        if(arr[i]== 0):
            for j in range(i+1,len(arr)):
                if(stop == False):
                    if(arr[j]>0):
                        arr[i]= arr[j]
                        arr[j] = 0
                        stop = True
    return(arr)
                        
def reevaluate_centroid(i):
    
    global after_sort
    
    sum_x = 0
    sum_y = 0
    cnt = 0
    for j in range(0,len(after_sort[i])):
        if(after_sort[i][j] > 0):
            sum_x = sum_x + data.lgt[data.ID == after_sort[i][j]].values[0]
            sum_y = sum_y + data.lat[data.ID == after_sort[i][j]].values[0]
            cnt = cnt + 1
    
    return([float(sum_x)/cnt,float(sum_y)/cnt])
    
def my_func1():
    
    global sort_clust_centre
    global after_sort
    global data
    global maxcol
    global clusters
    
    clust_size = []    
    
    for idx in range(0,max(estimator.labels_)+1):
        chk = np.count_nonzero(after_sort[idx])
        clust_size.append(chk)
        
    clust_size = np.array(clust_size) 
    
    after_sort_inter = np.zeros((max(estimator.labels_)+1,maxcol))    
    sort_clust_inter = []
    for j in range(0,max(estimator.labels_)+1):
        idx = np.where(clust_size == max(clust_size))[0][0]
        sort_clust_inter.append(sort_clust_centre[idx])
        for k in range(0,maxcol):
            after_sort_inter[j][k] = after_sort[idx][k]
    
            clust_size[idx] = 0
    after_sort = after_sort_inter
    sort_clust_centre = sort_clust_inter
    
    
    extra_budget = len(data['lgt'])%clusters
    extra = 0
    if(extra_budget > 0):
        
        extra = 1
        extra_budget = extra_budget-1
    
           
   
    
    for i in range(0,max(estimator.labels_)+1):
        #print(i)
        while((np.count_nonzero(after_sort[i]) > ((len(data['lgt'])/clusters)+extra))):
    #        for m in range(0,9):
            min_dist_inter = 10000.0
            for j in range(0,maxcol):
                #print(j)
                #print(after_sort[i])
                if(after_sort[i][j] > 0):
                    loc_x = data.lgt[data.ID == after_sort[i][j]].values[0]
                    loc_y = data.lat[data.ID == after_sort[i][j]].values[0]
                    loc = (loc_y,loc_x)
        
                    for k in range(i+1,max(estimator.labels_)+1):
                        
                        cc_x = sort_clust_centre[k][0]
                        cc_y = sort_clust_centre[k][1]
                        cc = (cc_y,cc_x)
                        dst = geopy.distance.vincenty(loc, cc).km
                        #print(dst)
                        if(dst<min_dist_inter):
                            min_dist_inter = dst 
                            min_cc_id = k
                            pop_id = j
            #print(min_dist_inter)
    #        print(min_cc_id)
            #print(pop_id)
            stop = False
            for l in range(0,maxcol):
                if(stop == False):
                    if(after_sort[min_cc_id][l]== 0):
                        after_sort[min_cc_id][l] = after_sort[i][pop_id]
                        after_sort[i][pop_id] = 0
                        after_sort[i] = rearrange(after_sort[i])
                        stop = True
                
            sort_clust_centre[i] = reevaluate_centroid(i)
            
        clust_size = []    
        
        for idx in range(0,max(estimator.labels_)+1):
            chk = np.count_nonzero(after_sort[idx])
            clust_size.append(chk)
        
        
        after_sort_inter = np.zeros((max(estimator.labels_)+1,maxcol))    
        sort_clust_inter = []
        for j in range(0,i+1):
            for k in range(0,maxcol):
                after_sort_inter[j][k] = after_sort[j][k]
                clust_size[j] = 0
            sort_clust_inter.append(sort_clust_centre[j])
        clust_size = np.array(clust_size)   
        
           
        for j in range(i+1,max(estimator.labels_)+1):
            idx = np.where(clust_size == max(clust_size))[0][0]
            sort_clust_inter.append(sort_clust_centre[idx])
            for k in range(0,maxcol):
                after_sort_inter[j][k] = after_sort[idx][k]
            
            clust_size[idx] = 0
        after_sort = after_sort_inter
        sort_clust_centre = sort_clust_inter
        
        if(extra_budget > 0):
            extra = 1
            extra_budget = extra_budget-1 
        else:
            extra = 0
            

my_func1()  
      
clr = np.zeros(len(data['lgt']))   

for i in range(0,max(estimator.labels_)+1):
    for j in range(0,maxcol):  
        if(after_sort[i][j] > 0):
            clr[int(after_sort[i][j])-1] = i  
            
plt.figure()       
plt.scatter(data['lgt'], data['lat'], c=clr) 
plt.show() 
plt.close()  

df = pd.DataFrame(data)
df['group'] = clr 
##data.to_csv("D:/Hackathon/data_Sudhu1.csv", encoding="utf-8")
#
l = dict()
for i,row in df.iterrows():
    team = row['ID']
    group  = row['group']
    #print(team, group)
    if group in l:
        l[group].append(team)
    else:
        l[group] = [team]

l2 = [l[g] for g in l]
print(l2)

def cluster_distance(clstr):
    total_dst = 0
    cnt = 0
    cc_x = sort_clust_centre[clstr][0]
    cc_y = sort_clust_centre[clstr][1]
    cc = (cc_y,cc_x)
    for j in range(0,maxcol): 
        if(after_sort[clstr][j] > 0):
            loc_x = data.lgt[data.ID == after_sort[clstr][j]].values[0]
            loc_y = data.lat[data.ID == after_sort[clstr][j]].values[0]
            loc = (loc_y,loc_x)
            dst = geopy.distance.vincenty(loc, cc).km
            total_dst = total_dst + dst
            cnt = cnt + 1
    return(total_dst/cnt)
            
def worst_cluster_distance(clstr):
    max_dst = 0
    max_dst_idx = 0
    cc_x = sort_clust_centre[clstr][0]
    cc_y = sort_clust_centre[clstr][1]
    cc = (cc_y,cc_x)
    for j in range(0,maxcol): 
        if(after_sort[clstr][j] > 0):
            loc_x = data.lgt[data.ID == after_sort[clstr][j]].values[0]
            loc_y = data.lat[data.ID == after_sort[clstr][j]].values[0]
            loc = (loc_y,loc_x)
            dst = geopy.distance.vincenty(loc, cc).km
            if(max_dst < dst):
                max_dst = dst
                max_dst_idx = j
    return max_dst,max_dst_idx
    
    
def reassign(clstr):
    global after_sort
    max_dst = 0
    pop_id = 0
    cc_x = sort_clust_centre[clstr][0]
    cc_y = sort_clust_centre[clstr][1]
    cc = (cc_y,cc_x)
    for j in range(0,maxcol):
        if(after_sort[clstr][j] > 0):
            loc_x = data.lgt[data.ID == after_sort[clstr][j]].values[0]
            loc_y = data.lat[data.ID == after_sort[clstr][j]].values[0]
            loc = (loc_y,loc_x)
            dst = geopy.distance.vincenty(loc, cc).km
            if(dst>max_dst):
                max_dst = dst
                pop_id = j
    
    loc_x = data.lgt[data.ID == after_sort[clstr][pop_id]].values[0]
    loc_y = data.lat[data.ID == after_sort[clstr][pop_id]].values[0]
    loc = (loc_y,loc_x)               
    
    min_dst = 10000
    min_cc_id = 0
    for i in range(0,max(estimator.labels_)+1):
        if(i != clstr):
                cc_x = sort_clust_centre[i][0]
                cc_y = sort_clust_centre[i][1]
                cc = (cc_y,cc_x)
                dst = geopy.distance.vincenty(loc, cc).km
                if(dst<min_dst):
                    min_dst = dst
                    min_cc_id = i
                    
    stop = False
    for l in range(0,maxcol):
        if(stop == False):
            if(after_sort[min_cc_id][l]== 0):
                after_sort[min_cc_id][l] = after_sort[clstr][pop_id]
                after_sort[clstr][pop_id] = 0
                after_sort[clstr] = rearrange(after_sort[clstr])
                stop = True
            

def get_team(clstr):  
    global after_sort
    
    cc_x = sort_clust_centre[clstr][0]
    cc_y = sort_clust_centre[clstr][1]
    cc = (cc_y,cc_x)
    
    min_dst = 10000
    min_cc_id = 0
    pop_id = 0
    
    for i in range(0,max(estimator.labels_)+1):
        if(i != clstr):
            for j in range(0,maxcol):
                if(after_sort[i][j] > 0):
                    loc_x = data.lgt[data.ID == after_sort[i][j]].values[0]
                    loc_y = data.lat[data.ID == after_sort[i][j]].values[0]
                    loc = (loc_y,loc_x)  
                    dst = geopy.distance.vincenty(loc, cc).km
                    if(dst<min_dst):
                        min_dst = dst
                        min_cc_id = i
                        pop_id = j
                    
    stop = False
    for l in range(0,maxcol):
        if(stop == False):
            if(after_sort[clstr][l]== 0):
                after_sort[clstr][l] = after_sort[min_cc_id][pop_id]
                after_sort[min_cc_id][pop_id] = 0
                after_sort[min_cc_id] = rearrange(after_sort[min_cc_id])
                stop = True
                
    return  min_cc_id               
  


for i in range(0,16):
    bad_cluster_dist = 0
    bad_index_dist = 0
    bad_cluster_inter = 0
    bad_index_dist_inter = 0
    for i in range(0,max(estimator.labels_)+1):
        bad_cluster_inter,bad_index_dist_inter = worst_cluster_distance(i)
        if(bad_cluster_inter>bad_cluster_dist):
            
                bad_cluster_dist = bad_cluster_inter
                bad_index_dist = i
#    print(bad_index_dist)
#    print(bad_cluster_dist)  
    reassign(bad_index_dist)
    lowteams = True
    highteams = True
    break_cnt = 0
    while ((lowteams == True) or (highteams == True)):
        
        lowteams = False
        highteams = False 
        if(np.count_nonzero(after_sort[bad_index_dist]) < 16):
            min_cc_id = get_team(bad_index_dist)
            bad_index_dist = min_cc_id
            lowteams = True
        
        for i in range(0,max(estimator.labels_)+1):
            if(np.count_nonzero(after_sort[i]) > 20):
                reassign(i)
                highteams = True
        break_cnt = break_cnt + 1
        #print(break_cnt)
        if(break_cnt > 5):
            break
    
        

clr = np.zeros(len(data['lgt']))   

for i in range(0,max(estimator.labels_)+1):
    for j in range(0,maxcol):  
        if(after_sort[i][j] > 0):
            clr[int(after_sort[i][j])-1] = i  
            
plt.figure()       
plt.scatter(data['lgt'], data['lat'], c=clr) 
plt.show() 



df = pd.DataFrame(data)
df['group'] = clr
##data.to_csv("D:/Hackathon/data_Sudhu1.csv", encoding="utf-8")
#
l = dict()
for i,row in df.iterrows():
    team = row['ID']
    group  = row['group']
    #print(team, group)
    if group in l:
        l[group].append(team)
    else:
        l[group] = [team]

l2 = [l[g] for g in l]
print(l2)





plt.close()
















#stop = False    
#for i in range(0,max(estimator.labels_)+1):
#    if(stop == False):
#        if((np.count_nonzero(after_sort[i]) > 20) or (np.count_nonzero(after_sort[i]) < 16)):
#            my_func1()
#            stop = True
#            
#clr = np.zeros(len(data['lgt']))   
#
#for i in range(0,max(estimator.labels_)+1):
#    for j in range(0,maxcol):  
#        if(after_sort[i][j] > 0):
#            clr[int(after_sort[i][j])-1] = i  
#            
#plt.figure()       
#plt.scatter(data['lgt'], data['lat'], c=clr) 
#plt.show()             
        
     

        