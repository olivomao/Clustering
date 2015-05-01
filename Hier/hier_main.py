# ---------- packages
import numpy as np
from numpy.linalg import norm
import pdb # debug

import csv # load data

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.spatial.distance as dt
import scipy.cluster.hierarchy as hr

from sklearn.metrics import silhouette_samples, silhouette_score # evaluation metric -- win ver
#from sklearn.metrics.cluster import silhouette_samples, silhouette_score # evaluation metric -- old ver

import argparse

import os

# ---------- read or prepare parameters -- linux
#print('read or prepare parameters')
#pa = argparse.ArgumentParser()
#pa.add_argument('Data_loc', help='Data location')
#pa.add_argument('N', help='# of samples')
#pa.add_argument('M', help='# of features')
#pa.add_argument('Case_loc', help='location to store output files')
#pa.add_argument('Pdist', help='pdist distance metric e.g. euclidean')
#pa.add_argument('Linkage_Method', help='Linkage_Method e.g. centroid')
#pa.add_argument('FCluster_Criterion', help='FCluster_Criterion e.g. distance')
#args = pa.parse_args()

#Data_loc = args.Data_loc
#N = int(args.N)
#M = int(args.M)
#Case_loc = args.Case_loc
#Pdist = args.Pdist
#Linkage_Method = args.Linkage_Method
#FCluster_Criterion = args.FCluster_Criterion

# ---------- read or prepare parameters -- win

# parameters
fn = 'Donghui_Kuang_RNAseq_raw_counts_12nov2014.txt'
#fn = 'matt_and_jerry_merged_rpm_26aug2014.Original.txt'
Data_loc = '../Data/'+fn

N = 68 #68
M = 300
Pdist = 'cityblock' #'euclidean'
Linkage_Method = 'single'
FCluster_Criterion = 'distance'
min_n_cluster = 3
max_n_cluster = 10

Case_loc = ('../Cases/Method_Hier_Clustering/DataSize_%d_%d/Pdist_'+Pdist+ \
           '/Linkage_Method_'+Linkage_Method+'/FCluster_Criterion_'+\
           FCluster_Criterion)%(N,M)

#pdb.set_trace()

if not os.path.exists(Case_loc):
    os.makedirs(Case_loc)

read_mode = 'r'
rand_seed = 10 # for reproducibility
y_space = 2

#pdb.set_trace()

# ---------- functions
def load_data(Data_loc, read_mode):
    col_labels = list()
    row_labels = list()
    data_matrix = list()
    
    with open(Data_loc, read_mode) as f:
        lines = csv.reader(f, delimiter='\t')
        i=0
        for row in lines:
            #pdb.set_trace()
            if i==0:
                col_labels = row[1:N+1] # col >=1 & <N+1
            elif i>M:
                break
            else:
                row_labels = row_labels + row[0].split() # row
                data_matrix = data_matrix + [row[1:N+1]] # matrix
            i=i+1
            
    return [row_labels, col_labels, data_matrix]

# ---------- main procedure
# read data & preprocessing
[RNA_labels, Sample_labels, Data_matrix] = load_data(Data_loc, read_mode)
#pdb.set_trace()
Data_matrix = np.asarray(Data_matrix, dtype=np.float64) #list to array
Data_matrix = Data_matrix.transpose() # N (samples) by M (features), row_labels=Sample_labels; col_labels=RNA_labels

#cut off (e.g. RNA 6~7 reads: ignore)
#N/A

#normalization
for i in range(len(Data_matrix)):
    if sum(Data_matrix[i])!=0: # to avoid 'nan'
        Data_matrix[i] = Data_matrix[i]/sum(Data_matrix[i])

pdb.set_trace()
#hierarchical clustering
Dist_matrix = dt.pdist(Data_matrix, Pdist)
#for debug:
Dist_matrix = dt.squareform(Dist_matrix) #, checks=True) --> returns a (68,68) matrix; needed for other methods of linkage
#check its histogram:
#(h,b)=np.histogram(Dist_matrix)
#(f_hist, axis_hist)=plt.subplots()
#axis_hist.plot(b[1:], h)
#f_hist.show()

#pdb.set_trace()
#Hier_clustering = hr.linkage(Dist_matrix) #, method='centroid') #, method=Linkage_Method, metric=Pdist)
Hier_clustering = hr.linkage(Dist_matrix, method=Linkage_Method, metric=Pdist)

dendro = hr.dendrogram(Hier_clustering)

#plt.show()
#try to get current axes & modify & save figures
ax_dendro = plt.gca()
fig_dendro = plt.gcf()
fig_dendro.savefig(Case_loc+'/'+'fig_dendrogram.png')

#visualization
#pdb.set_trace()

n_cluster_list = list()
for ith_t in Hier_clustering[:,2]:
    cluster_labels = hr.fcluster(Hier_clustering, ith_t, criterion=FCluster_Criterion)
    cluster_labels = cluster_labels - 1 # start from 0
    n_clusters = cluster_labels.max()+1 # cluster index = {0,...,N-1} --> N clusters
    n_cluster_list = n_cluster_list + [[ith_t, n_clusters]]

n_cluster_list = np.asarray(n_cluster_list, dtype=np.float64)
n_cluster_list = n_cluster_list[np.argsort(n_cluster_list[:,1])]

#pdb.set_trace()
n_cluster_list_2 = list() # choose t that brings n_clusters in [min_n_clusters, max_n_clusters]
prev_n_cluster = 0
for row in n_cluster_list:
    curr_n_cluster = int(row[1])
    if curr_n_cluster>=min_n_cluster and curr_n_cluster<=max_n_cluster \
       and curr_n_cluster != prev_n_cluster:
        prev_n_cluster = curr_n_cluster
        n_cluster_list_2 = n_cluster_list_2 + [[row[0], curr_n_cluster, 0]] # 3rd item is to record cluster metric performance later

n_cluster_list_2 = np.asarray(n_cluster_list_2)

#pdb.set_trace()

i = 0
for row in n_cluster_list_2:
    
    curr_t = row[0]
    cluster_labels = hr.fcluster(Hier_clustering, curr_t, criterion=FCluster_Criterion)
    cluster_labels = cluster_labels - 1 # start from 0
    n_clusters = int(row[1])
    print('process cluster_num = %d' % n_clusters)

    silhouette_avg = silhouette_score(Data_matrix, cluster_labels, metric=Pdist)
    n_cluster_list_2[i,2] = silhouette_avg
    i = i+1
    
    sample_silhouette_values = silhouette_samples(Data_matrix, cluster_labels, metric=Pdist)

    #for certain cutoff t (i.e. cluster size), plot clusters' silhouette values' distribution
    #for certain cutoff t (i.e. cluster size), plot each cluster's point feature distribution
    
    min_silhouette_value = min(sample_silhouette_values)
    start_x = min_silhouette_value - 0.1

    ax1 = plt.subplot2grid((n_clusters,2), (0,0), rowspan=n_clusters)
    ax1.set_xlim([start_x,1])
    ax1.set_ylim([0, len(Data_matrix)+ (n_clusters+1)*y_space])
    #ax1.set_title('The silhouette plot for the various clusters')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.grid(b=True, which='both', axis='both')

    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

    ax_list = list()
    ax_list.append(ax1)

    Case_loc_sub = ((Case_loc+'/'+'ClusterNum%d')%n_clusters)
    if not os.path.exists(Case_loc_sub):
        os.makedirs(Case_loc_sub)

    y_lower = y_space

    for ith_cluster in range(n_clusters):
        print('--> %d th cluster' % ith_cluster)
        ax_tmp = plt.subplot2grid((n_clusters, 2), (n_clusters-1-ith_cluster, 1))  # 0-th curve at bottom
        ax_list.append(ax_tmp)

        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels==ith_cluster]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(ith_cluster) / n_clusters)

        #if size_cluster_i == 1:
        #    pdb.set_trace()
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(0.5, y_lower + 0.5*size_cluster_i,
                 str('%d-th, %d samples' %(ith_cluster, size_cluster_i)), color=color)
        y_lower = y_upper + y_space

        #right subplots
        ith_cluster_data =  Data_matrix[cluster_labels==ith_cluster]

        #pdb.set_trace()
        #mean or centroid point
        tmp_vec = np.mean(ith_cluster_data, axis=0)
        tmp_mat = ith_cluster_data - tmp_vec
        tmp_mat = np.array([norm(v) for v in tmp_mat])
        min_sample_idx = np.argmin(tmp_mat)
        max_sample_idx = np.argmax(tmp_mat)

        ith_cluster_centroid_x = range(0, len(tmp_vec))
        ith_cluster_centroid_y = tmp_vec

        ith_cluster_min_sample_curve_x = range(0, len(ith_cluster_data[min_sample_idx]))
        ith_cluster_min_sample_curve_y = ith_cluster_data[min_sample_idx]

        ith_cluster_max_sample_curve_x = range(0, len(ith_cluster_data[max_sample_idx]))
        ith_cluster_max_sample_curve_y = ith_cluster_data[max_sample_idx]

        ax_tmp.set_xlim([-10, len(ith_cluster_data[0])+10])
        
        ax_tmp.plot(ith_cluster_centroid_x,
                    ith_cluster_centroid_y,
                    color='black', linestyle='--')

        ax_tmp.plot(ith_cluster_min_sample_curve_x,
                    ith_cluster_min_sample_curve_y,
                    color='red', linestyle='--')

        ax_tmp.plot(ith_cluster_max_sample_curve_x,
                    ith_cluster_max_sample_curve_y,
                    color=color, linestyle='--')

        if ith_cluster==0: # bottom
            ax_tmp.set_xlabel('RNA-seq index')
        else:
            ax_tmp.set_xticks([])

        max_ytick = max(max(ith_cluster_centroid_y),
                        max(ith_cluster_min_sample_curve_y),
                        max(ith_cluster_max_sample_curve_y))
        ax_tmp.set_yticks([0, max_ytick])
        ax_tmp.yaxis.tick_right()
        if ith_cluster==n_clusters-1: # top
            ax_tmp.set_title('RNA-seq distr')

        #save larger version
        #pdb.set_trace()
        #fig_tmp2, ax_tmp2 = plt.subplots()

        #ax_tmp2.plot(ith_cluster_centroid_x,
        #            ith_cluster_centroid_y,
        #            color='black', linestyle='--')

        #ax_tmp2.plot(ith_cluster_min_sample_curve_x,
        #            ith_cluster_min_sample_curve_y,
        #            color='red', linestyle='--')

        #ax_tmp2.plot(ith_cluster_max_sample_curve_x,
        #            ith_cluster_max_sample_curve_y,
        #            color=color, linestyle='--')
        
        #fig_tmp2.savefig(Case_loc_sub+'/fig_%dth_cluster_RNA_dist'%(ith_cluster))

    plt.savefig(Case_loc_sub+'/'+'fig_silhouette_distribution.png')

#plot silhouette performance regarding to different cluster size values
pdb.set_trace()
fig_t, ax_t = plt.subplots()
v_cluster_size = n_cluster_list_2[:,1]
v_silhouette_avg = n_cluster_list_2[:,2]
ax_t.set_title('The avg silhouette for various cluster sizes')
ax_t.set_xlabel('Cluster Size')
ax_t.set_ylabel('The avg silhouette coefficient values')
ax_t.grid(b=True, which='both', axis='both')
ax_t.plot(v_cluster_size, v_silhouette_avg, marker='o')
fig_t.savefig(Case_loc+'/'+'fig_avg_silhouette_vs_cluster_size.png')

print('Ready to exit')
#pdb.set_trace()
