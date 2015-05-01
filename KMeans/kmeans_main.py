# packages
import numpy as np
from numpy.linalg import norm
import pdb # debug

import csv # load data

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans # clustering method
from sklearn.metrics import silhouette_samples, silhouette_score # evaluation metric
#from sklearn.metrics.cluster import silhouette_samples, silhouette_score # evaluation metric

import os


# parameters
fn = 'Donghui_Kuang_RNAseq_raw_counts_12nov2014.txt'
#fn = 'matt_and_jerry_merged_rpm_26aug2014.Original.txt'
file_name = '../Data/'+fn
read_mode = 'r'
N = 100
max_r = 30 # read max_r RNAs / max_r=10~100 for code debug purpose
min_n_clusters = 3
max_n_clusters = 10
range_n_clusters = range(min_n_clusters, max_n_clusters, 1)#[2, 3, 4, 5, 6] # try different number of clusters
rand_seed = 10 # for reproducibility
y_space = 2 # subplot1
subplot_height = 18
subplot_width = 18

Case_loc = ('../Cases/Method_Kmeans_Clustering/File_%s/DataSize_%d_%d')%(fn, N,max_r)
if not os.path.exists(Case_loc):
    os.makedirs(Case_loc)

# functions
def load_data(file_name, read_mode):
    col_labels = list()
    row_labels = list()
    data_matrix = list()
    
    with open(file_name, read_mode) as f:
        lines = csv.reader(f, delimiter='\t')
        i=0
        for row in lines:
            if i==0:
                col_labels = row[1:N+1] # col
            elif i>max_r:
                break
            else:
                row_labels = row_labels + row[0].split() # row
                data_matrix = data_matrix + [row[1:N+1]] # matrix
            i=i+1
            
    return [row_labels, col_labels, data_matrix]

pdb.set_trace()

# read data & preprocessing
[RNA_labels, Sample_labels, Data_matrix] = load_data(file_name, read_mode)
Data_matrix = np.asarray(Data_matrix, dtype=np.float64) #list to array
Data_matrix = Data_matrix.transpose() # row_labels=Sample_labels; col_labels=RNA_labels

#cut off (e.g. RNA 6~7 reads: ignore)
#N/A

#normalization
for i in range(len(Data_matrix)):
    if sum(Data_matrix[i])!=0: # to avoid 'nan' which causes problem in KMeans
        #pdb.set_trace()
        Data_matrix[i] = Data_matrix[i]/sum(Data_matrix[i])

#k-means clustering
all_clusterers = list()
for n_clusters in range_n_clusters:
    #pdb.set_trace()
    clusterer = KMeans(n_clusters=n_clusters, random_state=rand_seed)
    #clusterer = KMeans(k=n_clusters, random_state=rand_seed) #for old version
    clusterer.fit(Data_matrix)
    all_clusterers.append(clusterer)

#visualization 1 (high-dimen clustering)
all_silhouette_avg = list()
for clusterer in all_clusterers:
    #pdb.set_trace()

    n_clusters = len(clusterer.cluster_centers_)
    print('process cluster_num=%d' % n_clusters)
    cluster_labels = clusterer.labels_
    
    silhouette_avg = silhouette_score(Data_matrix, cluster_labels)
    sample_silhouette_values = silhouette_samples(Data_matrix, cluster_labels)
    min_silhouette_value = min(sample_silhouette_values)
    start_x = min_silhouette_value - 0.1
    #pdb.set_trace()
    all_silhouette_avg.append(silhouette_avg)

    #pdb.set_trace()
    ax_list = list()
    ax1 = plt.subplot2grid((n_clusters,2), (0,0), rowspan=n_clusters)
    ax_list.append(ax1)
    for i in range(n_clusters):
        ax_tmp = plt.subplot2grid((n_clusters,2), (n_clusters-1-i,1)) # 0-th curve at bottom
        ax_list.append(ax_tmp)

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.set_size_inches(subplot_height,subplot_width)

    ax1.set_xlim([start_x,1])
    ax1.set_ylim([0, len(Data_matrix)+ (n_clusters+1)*y_space])

    Case_loc_sub = ((Case_loc+'/'+'ClusterNum%d')%n_clusters)
    if not os.path.exists(Case_loc_sub):
        os.makedirs(Case_loc_sub)

    #left subplot
    y_lower = y_space
    for i in range(n_clusters):
        print('-->process %dth cluster'%i)
        #pdb.set_trace()
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels==i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        #print('%d-th cluster, %d samples'%(i, size_cluster_i))
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        #ax1.text(-0.8, y_lower + 0.5*size_cluster_i, str('%d-th cluster' % i))
        #ax1.text(-0.05, y_lower + 0.5*size_cluster_i, str(i))
        ax1.text(0.5, y_lower + 0.5*size_cluster_i, str('%d-th, %d samples' %(i, size_cluster_i)), color=color)
        #if i>0:
        #    ax1.axhline(y=y_lower - y_space/2, color='grey', linestyle='--')
        y_lower = y_upper + y_space

        #right subplots
        #pdb.set_trace()
        
        ith_cluster_data =  Data_matrix[cluster_labels==i]
        ith_cluster_centroid_curve_x = range(0, clusterer.cluster_centers_.shape[1])
        ith_cluster_centroid_curve_y = clusterer.cluster_centers_[i]

        #dist from centroid
        tmp_mat = ith_cluster_data - ith_cluster_centroid_curve_y
        tmp_mat = np.array([norm(v) for v in tmp_mat])
        min_sample_idx = np.argmin(tmp_mat)
        max_sample_idx = np.argmax(tmp_mat)

        ith_cluster_min_sample_curve_x = range(0, len(ith_cluster_data[min_sample_idx]))
        ith_cluster_min_sample_curve_y = ith_cluster_data[min_sample_idx]

        ith_cluster_max_sample_curve_x = range(0, len(ith_cluster_data[max_sample_idx]))
        ith_cluster_max_sample_curve_y = ith_cluster_data[max_sample_idx]
                
        ith_ax = ax_list[i+1] #exluding ax1
        ith_ax.set_xlim([-10, clusterer.cluster_centers_.shape[1]+10])
        #ith_ax.set_ylim([-0.0001,1.0001])
        #ith_ax.plot(ith_cluster_centroid_curve_x,
        #            ith_cluster_centroid_curve_y,
        #            color='red')
        #ith_ax.plot(ith_cluster_one_sample_curve_x,
        #           ith_cluster_one_sample_curve_y,
        #            color=color, linestyle='--')

        ith_ax.plot(ith_cluster_centroid_curve_x,
                    ith_cluster_centroid_curve_y,
                    color='black', linestyle='--')

        ith_ax.plot(ith_cluster_min_sample_curve_x,
                    ith_cluster_min_sample_curve_y,
                    color='red', linestyle='--')

        ith_ax.plot(ith_cluster_max_sample_curve_x,
                    ith_cluster_max_sample_curve_y,
                    color=color, linestyle='--')
        
        if i==0:
            ith_ax.set_xlabel('RNA-seq index')
        else:
            ith_ax.set_xticks([])

        max_ytick = max(max(ith_cluster_centroid_curve_y),
                        max(ith_cluster_min_sample_curve_y),
                        max(ith_cluster_max_sample_curve_y))
        ith_ax.set_yticks([0, max_ytick])
        ith_ax.yaxis.tick_right()

        if i==n_clusters-1:
            ith_ax.set_title('RNA-seq dist')
        
        #pdb.set_trace()
        
    ax1.set_title('The silhouette plot for the various clusters')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.grid(b=True, which='both', axis='both')

    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

    #right subplot

    #
    #pdb.set_trace()
    #plt.suptitle(('Silhouette analysis for KMeans clustering on sample data '
    #              'with n_clusters = %d' % n_clusters), fontsize=14, fontweight='bold')
    #plt.show()
    plt.savefig(Case_loc_sub+'/'+'fig_silhouette_distribution.png')
    #plt.savefig('fig_max_r_%d_cluster_num_%d.png' %(max_r, n_clusters))
    #input('next fig?')
    
#visualization 2 (how # of clusters will influence silhouette avg score)
fig, ax = plt.subplots()
x=np.arange(min_n_clusters, max_n_clusters, 1)
z = np.asarray(all_silhouette_avg)

ax.plot(x, z, marker='o')
ax.set_title('The avg silhouette for various cluster sizes')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('The avg silhouette coefficient values')
ax.grid(b=True, which='both', axis='both')
#plt.show()
fig.savefig(Case_loc+'/'+'fig_avg_silhouette_vs_cluster_size.png')

print('Ready to exit')
