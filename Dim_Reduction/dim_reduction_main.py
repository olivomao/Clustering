# ---------- packages
import numpy as np
from numpy.linalg import norm
import pdb # debug

import csv # load data

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

import scipy.spatial.distance as dt
import scipy.cluster.hierarchy as hr
from sklearn.cluster import KMeans # clustering method

from time import time

import argparse

import os

from sklearn import manifold

from sklearn.decomposition import PCA, SparsePCA

#import seaborn as sns # for heatmap purpose

import warnings
#warnings.filterwarnings("ignore") #diable warning
#from sklearn.utils.testing import assert_true
#from sklearn.utils.testing import assert_false

# ---------- packages (self-written)
import sys
sys.path.insert(1, '../Drawing/')
exportHeatmap = True
if exportHeatmap == True:
    import draw_heatmap, draw_heatmap2


# ---------- read or prepare parameters -- win
isWin = 1 #0: linux old python version
if isWin==1:
    from sklearn.metrics import silhouette_samples, silhouette_score # evaluation metric -- win ver
else:
    from sklearn.metrics.cluster import silhouette_samples, silhouette_score # evaluation metric -- linux ver

Data_loc = '../Data/Donghui_Kuang_RNAseq_raw_counts_12nov2014.txt' # 68*24015
fn = 'Data1'
#Data_loc = '../Data/matt_and_jerry_merged_rpm_26aug2014.Original.txt' # 53*2????
#fn = 'Data2' #for output location
N = 68
M = 30000
bins = 10 # used to check histogram of sample reads
if M>20000:
    if fn == 'Data1':
        BadSampleThreshold = 5e5 #5e5 #np.arange(5e5, 1e6, 1e5) for Data1, 0 for Data2
    elif fn == 'Data2':
        BadSampleThreshold = 0
    else:
        pdb.set_trace()
else: #for debug purpose
    BadSampleThreshold = 2000

RemoveBadSamples = True # Remove from Data_Matrix the Samples whose total reads are below BadSampleThreshold

reduce_data_dim = False # reduce data dimension before clustering
if reduce_data_dim == True:
    reduce_data_dim_method = 'pca' #'pca' #'isomap' 'tsne' 'lle' 'pca'
    if reduce_data_dim_method == 'lle':
        reduce_data_dim_lle_method = 'modified' # 'standard', 'ltsa', 'hessian', 'modified'
    elif reduce_data_dim_method == 'isomap':
        #For ISO map
        reduce_data_dim_n_neighbors = 10 # k-isomap
    else:
        #pdb.set_trace()
        print('unknown reduce data dim method -- %s' % reduce_data_dim_method)
    reduce_data_dim_n_components = 2 #2 3
    reduce_data_dim_str = ('[ReduceDim_%s_%s_dim%d]')%(reduce_data_dim, reduce_data_dim_method, reduce_data_dim_n_components)
else:
    reduce_data_dim_str = ('[ReduceDim_%s]')%(reduce_data_dim)

clustering_method = 'kmeans' # kmeans, kmedoid, hier
if clustering_method == 'kmeans':
    #pdb.set_trace()
    Pdist = 'euclidean'
    clustering_method_str = 'kmeans'
elif clustering_method == 'hier':
    #pdb.set_trace()
    Pdist = 'euclidean' #'euclidean' 'cityblock'
    Linkage_Method = 'ward' #'complete' (looks seperated) #'average' #'single' 'ward'
    FCluster_Criterion = 'distance' #'inconsistent'
    #Pdist = 'euclidean' #also to calculate silhouette
    #clustering_method_str = 'hier_[Dist_%s_Linkage_%s_FC_%s]'%(Pdist, Linkage_Method, FCluster_Criterion)
    clustering_method_str = 'hier'
elif clustering_method == 'kmedoid':
    pdb.set_trace()
    Pdist = 'cityblock' #or Manhattan distance
    clustering_method_str = 'kmedoid'
else:
    pdb.set_trace()
    print('unknown clustering method')

min_n_cluster = 3
max_n_cluster = 3
range_n_clusters = range(min_n_cluster, max_n_cluster+1, 1)

dm_method = 'pca' #'pca' #'isomap' 'tsne' 'lle' 'pca' 'sparse-pca'
if dm_method == 'lle':
    lle_method = 'modified' # 'standard', 'ltsa', 'hessian', 'modified'
elif dm_method == 'isomap':
    #For ISO map
    n_neighbors = 10 # k-isomap
elif dm_method == 'pca':
    dm_method = 'pca' #dummy code
elif dm_method == 'sparse-pca':
    alpha = 0.001
    ridge_alpha = 0.05
    dm_method = 'sparse-pca'
else:
    pdb.set_trace()
    print('unknown dm method -- %s'%dm_method)
n_components = 2 #2 3

Case_loc = ('../Cases/Clustering_%s_%s_DM_%s/%s/DataSize_%d_%d/')%(clustering_method_str, reduce_data_dim_str, dm_method, \
                                                               fn, N, M)
print('case loc: %s'%Case_loc)

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

# [GoodSamples_index, BadSamples_index] = FilterBadSamples(Data_matrix, BadSampleThreshold)
def FilterBadSamples(Data_matrix, BadSampleThreshold):
    #pdb.set_trace()
    #Data_matrix_GoodSamples = [row for row in Data_matrix if sum(row)>BadSampleThreshold]
    #Data_matrix_BadSamples = [row for row in Data_matrix if sum(row)<=BadSampleThreshold]
    #return [Data_matrix_GoodSamples, Data_matrix_BadSamples]
    GoodSamples_index  = [ i for i in range(len(Data_matrix)) if sum(Data_matrix[i])>BadSampleThreshold ]
    BadSamples_index = [ i for i in range(len(Data_matrix)) if sum(Data_matrix[i])<=BadSampleThreshold ]
    return [GoodSamples_index, BadSamples_index]

# function -- dimension reduction using pca
def dm_pca(Data_matrix, n_components):
    #pdb.set_trace()
    print('pca starts')
    t0 = time()
    Y = PCA(n_components=n_components).fit_transform(Data_matrix)
    t1 = time()
    print('\tpca finishes with %.2g sec' % (t1-t0))
    return Y

# function -- dimension reduction using pca
def dm_sparse_pca(Data_matrix, n_components, alpha, ridge_alpha):
    pdb.set_trace()
    print('sparse pca starts')
    t0 = time()
    #Y = SparsePCA(n_components=n_components).fit_transform(Data_matrix)
    Y = SparsePCA(n_components, alpha=alpha, ridge_alpha=ridge_alpha).fit_transform(np.asarray(Data_matrix))
    #assert_false(np.any(np.isnan(Y)))
    t1 = time()
    print('\tsparse pca finishes with %.2g sec' % (t1-t0))
    return Y

# function -- dimension reduction using lle ('standard', 'ltsa', 'hessian', 'modified')
def dm_lle(Data_matrix, n_neighbors, n_components, method):
    pdb.set_trace()
    print('lle - %s starts' % method)
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto', method=method).fit_transform(Data_matrix)
    t1 = time()
    print('lle - %s finishes with %.2g sec' % (method, (t1-t0)))
    return Y

# function -- dimension reduction using tsne
def dm_tsne(Data_matrix, n_components):
    #pdb.set_trace()
    print('tsne starts')
    t0 = time()
    Y = manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(Data_matrix)
    t1 = time()
    print('tsne finishes with %.2g sec' % (t1-t0))
    return Y
    
# function -- dimension reduction using isomap
def dm_isomap(Data_matrix, n_neighbors, n_components):
    #pdb.set_trace()
    print('isomap starts')
    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(Data_matrix)
    t1 = time()
    print('isomap finishes with %.2g sec' % (t1-t0))
    return Y
    
# function -- clustering using kmeans
def clst_kmeans(Data_matrix, n_clusters):
    #pdb.set_trace()
    print('kmeans starts')
    t0 = time()
    if isWin==1:
        clusterer = KMeans(n_clusters=n_clusters, random_state=rand_seed)
    else:
        clusterer = KMeans(k=n_clusters, random_state=rand_seed) #for linux
    clusterer.fit(Data_matrix)
    t1 = time()
    print('\tkmeans finishes with %.2g sec' % (t1-t0))

    #pdb.set_trace()
    cluster_labels = clusterer.labels_
    SD = clusterer.inertia_ #SD: sum of distortion =
                            #Sum of distances of samples to their closest cluster center
    color_matrix = np.zeros(len(cluster_labels)*4)
    color_matrix = color_matrix.reshape((len(cluster_labels), 4))
    for i in range(n_clusters):
        ith_cluster_color = cm.spectral(float(i) / n_clusters)
        color_matrix[cluster_labels==i] = ith_cluster_color

    return [cluster_labels, color_matrix, SD]

#export: dendrogram fig
def clst_hier(Data_matrix, Linkage_Method, Pdist, n_clusters):
    #pdb.set_trace()
    #hierarchical clustering
    Dist_matrix = dt.pdist(Data_matrix, Pdist)
    Dist_matrix = dt.squareform(Dist_matrix) #, checks=True) --> returns a square matrix; needed for other methods of linkage
    #check its histogram:
    #(h,b)=np.histogram(Dist_matrix)
    #(f_hist, axis_hist)=plt.subplots()
    #axis_hist.plot(b[1:], h)
    #f_hist.show()

    #pdb.set_trace()
    #Hier_clustering = hr.linkage(Dist_matrix) #, method='centroid') #, method=Linkage_Method, metric=Pdist)
    Hier_clustering = hr.linkage(Dist_matrix, method=Linkage_Method, metric=Pdist)

    #draw dendrogram
    dendro = hr.dendrogram(Hier_clustering)

    #plt.show()
    #try to get current axes & modify & save figures
    ax_dendro = plt.gca()
    fig_dendro = plt.gcf()
    #pdb.set_trace()
    fig_dendro.savefig(Case_loc+'fig_dendrogram.png')

    #pdb.set_trace()
    #n_cluster_list = list()
    tmp_n_clusters = 0
    for ith_t in Hier_clustering[:,2]:
        cluster_labels = hr.fcluster(Hier_clustering, ith_t, criterion=FCluster_Criterion)
        cluster_labels = cluster_labels - 1 # start from 0
        tmp_n_clusters = cluster_labels.max()+1 # cluster index = {0,...,N-1} --> N clusters
        if tmp_n_clusters == n_clusters:
            break
    
    if tmp_n_clusters == 0:
        print('unable to find %d clusters in clst_hier'%n_clusters)
        pdb.set_trace()

    color_matrix = np.zeros(len(cluster_labels)*4)
    color_matrix = color_matrix.reshape((len(cluster_labels), 4))
    for i in range(n_clusters):
        ith_cluster_color = cm.spectral(float(i) / n_clusters)
        color_matrix[cluster_labels==i] = ith_cluster_color
        
    SD = 0 #currently not found intertia for hier method

    return [cluster_labels, color_matrix, SD]
    

def clst_kmedoid(Data_matrix, n_clusters):
    pdb.set_trace()
    print('kmedoid starts')
    t0 = time()
    if isWin==1:
        print('pos1')
        #clusterer = KMeans(n_clusters=n_clusters, random_state=rand_seed)
    else:
        print('pos2')
        #clusterer = KMeans(k=n_clusters, random_state=rand_seed) #for linux
    #clusterer.fit(Data_matrix)
    t1 = time()
    print('kmedoid finishes with %.2g sec' % (t1-t0))

    #pdb.set_trace()
    #cluster_labels = clusterer.labels_
    #color_matrix = np.zeros(len(cluster_labels)*4)
    #color_matrix = color_matrix.reshape((len(cluster_labels), 4))
    for i in range(n_clusters):
        print('pos3')
        #ith_cluster_color = cm.spectral(float(i) / n_clusters)
        #color_matrix[cluster_labels==i] = ith_cluster_color

    return [cluster_labels, color_matrix] 

# function -- get silhouette information
def get_silhouette_info(Data_matrix, cluster_labels, distance_metric):
    #pdb.set_trace()
    sample_silhouette_values = silhouette_samples(Data_matrix, cluster_labels, metric=distance_metric)
    sample_silhouette_values = np.nan_to_num(sample_silhouette_values)
    silhouette_avg = np.mean(sample_silhouette_values) #silhouette_score(np.asarray(Data_matrix), cluster_labels, metric=distance_metric)
    return [sample_silhouette_values, silhouette_avg]


# function -- check histogram of sample reads
def check_histogram_of_sample_reads(Data_matrix, bins):
    pdb.set_trace()
    num_reads_list = [sum(Data_matrix[i]) for i in range(len(Data_matrix))]
    num_reads_list = np.asarray(num_reads_list)
    [hist, edge] = np.histogram(num_reads_list, bins=bins)
    plt.plot(edge[1:], hist)
    plt.show()

# function -- draw scatter
def fig_scatter_sub(Y, color_matrix,
                    ref_idx1, ref_idx2,
                    s0, marker0,
                    ax, is2D):
    Y_sub = [Y[i] for i in range(len(Y)) if i in ref_idx1 and i in ref_idx2]
    Y_sub = np.asarray(Y_sub)
    color_matrix_sub = [color_matrix[i] for i in range(len(Y)) if i in ref_idx1 and i in ref_idx2]
    color_matrix_sub = np.asarray(color_matrix_sub)
    if len(Y_sub)!= 0: # it's possible that ref_idx1 and ref_idx2 intersect to empty set
        if is2D==1:
            try:
                ax.scatter(Y_sub[:, 0], Y_sub[:, 1], s=s0, c=color_matrix_sub, marker=marker0, cmap=plt.cm.Spectral)
            except:
                print('fig_scatter_sub exception (is2D==1)')
                pdb.set_trace()
        else:
            try:
                ax.scatter(Y_sub[:, 0], Y_sub[:, 1], Y_sub[:, 2],
                           s=s0, c=color_matrix_sub, marker=marker0, cmap=plt.cm.Spectral)
            except:
                print('fig_scatter_sub exception (is2D==0)')
                pdb.set_trace()
    

def fig_scatter(Y, color_matrix, dm_method, n_components,
            Case_loc, n_clusters, RemoveBadSamples,
            GoodSamples_index, BadSamples_index, NonNegativeSilhouette_Index, NegativeSilhouette_Index):
    #pdb.set_trace()
    fig = plt.figure(figsize=(15, 8))
    if n_components == 2: #2D
        ax = fig.add_subplot(111)
        #good samples with default marker 'o'
        #Y_good = [Y[i] for i in range(len(Y)) if i in GoodSamples_index]
        #Y_good = np.asarray(Y_good)
        #color_matrix_good = [color_matrix[i] for i in range(len(Y)) if i in GoodSamples_index]
        #color_matrix_good = np.asarray(color_matrix_good)
        #ax.scatter(Y_good[:,0], Y_good[:,1], s=20, c=color_matrix_good, marker='o', cmap=plt.cm.Spectral)
        fig_scatter_sub(Y, color_matrix,
                    GoodSamples_index, NonNegativeSilhouette_Index,
                    50, 'o',
                    ax, 1)
        fig_scatter_sub(Y, color_matrix,
                    GoodSamples_index, NegativeSilhouette_Index,
                    150, 'o',
                    ax, 1)
        #bad samples with marker '>'
        #Y_bad = [Y[i] for i in range(len(Y)) if i in BadSamples_index]
        #Y_bad = np.asarray(Y_bad)
        #color_matrix_bad = [color_matrix[i] for i in range(len(Y)) if i in BadSamples_index]
        #color_matrix_bad = np.asarray(color_matrix_bad)
        #ax.scatter(Y_bad[:,0], Y_bad[:,1], s=20, c=color_matrix_bad, marker='>', cmap=plt.cm.Spectral)
        if len(BadSamples_index)>0:
            fig_scatter_sub(Y, color_matrix,
                    BadSamples_index, NonNegativeSilhouette_Index,
                    50, '>',
                    ax, 1)
            fig_scatter_sub(Y, color_matrix,
                    BadSamples_index, NegativeSilhouette_Index,
                    150, '>',
                    ax, 1)
    else: #3D
        Axes3D
        ax = fig.add_subplot(111, projection='3d')
        #ax = plt.gca()
        #ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=color_matrix, cmap=plt.cm.Spectral) #color #here Y is a (n_samples, n_components) matrix

        #good samples with default marker 'o'
        #Y_good = [Y[i] for i in range(len(Y)) if i in GoodSamples_index]
        #Y_good = np.asarray(Y_good)
        #color_matrix_good = [color_matrix[i] for i in range(len(Y)) if i in GoodSamples_index]
        #color_matrix_good = np.asarray(color_matrix_good)
        #ax.scatter(Y_good[:,0], Y_good[:,1], Y_good[:,2], c=color_matrix_good, marker='o', cmap=plt.cm.Spectral)

        fig_scatter_sub(Y, color_matrix,
                    GoodSamples_index, NonNegativeSilhouette_Index,
                    50, 'o',
                    ax, 0)
        fig_scatter_sub(Y, color_matrix,
                    GoodSamples_index, NegativeSilhouette_Index,
                    150, 'o',
                    ax, 0)
        
        #bad samples with marker '>'
        #Y_bad = [Y[i] for i in range(len(Y)) if i in BadSamples_index]
        #Y_bad = np.asarray(Y_bad)
        #color_matrix_bad = [color_matrix[i] for i in range(len(Y)) if i in BadSamples_index]
        #color_matrix_bad = np.asarray(color_matrix_bad)
        #ax.scatter(Y_bad[:,0], Y_bad[:,1], Y_bad[:,2], c=color_matrix_bad, marker='>', cmap=plt.cm.Spectral)

        if len(BadSamples_index)>0:
            fig_scatter_sub(Y, color_matrix,
                    BadSamples_index, NonNegativeSilhouette_Index,
                    50, '>',
                    ax, 0)
            fig_scatter_sub(Y, color_matrix,
                    BadSamples_index, NegativeSilhouette_Index,
                    150, '>',
                    ax, 0)

        ax.view_init(4, -72)
    plt.title(('%s_NoBadpoints_%s_n_clusters_%d_%dD')%(dm_method, RemoveBadSamples, n_clusters, n_components))
    #ax = plt.gca()
    #ax.xaxis.set_major_formatter('') #(NullFormatter())
    #ax.yaxis.set_major_formatter('') #(NullFormatter())
    #plt.axis('tight')
    if n_components == 3:
        plt.show()
    else:
        #plt.show()
        plt.savefig((Case_loc+'NoBadpoints_%s_n_clusters_%d_%dD'+'.png')%(RemoveBadSamples, n_clusters, n_components))

def fig_silhouette_distribution(Data_matrix,
                                sample_silhouette_values, silhouette_avg,
                                n_clusters, cluster_labels,
                                Case_loc):
    Data_matrix = np.asarray(Data_matrix)
    #pdb.set_trace()
    min_silhouette_value = min(sample_silhouette_values)
    start_x = min_silhouette_value - 0.1

    ax1 = plt.subplot2grid((n_clusters,2), (0,0), rowspan=n_clusters)

    ax1.set_xlim([start_x,1])
    ax1.set_ylim([0, len(Data_matrix)+ (n_clusters+1)*y_space])
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.grid(b=True, which='both', axis='both')
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

    y_lower = y_space
    for ith_cluster in range(n_clusters):
        #print('--> %d th cluster' % ith_cluster)
        ax_tmp = plt.subplot2grid((n_clusters, 2), (n_clusters-1-ith_cluster, 1))  # 0-th curve at bottom
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels==ith_cluster]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(ith_cluster) / n_clusters)

        #left subplots
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(0.5, y_lower + 0.5*size_cluster_i,
                 str('%d-th, %d samples' %(ith_cluster, size_cluster_i)), color=color)
        y_lower = y_upper + y_space

        #right subplots
        #pdb.set_trace()
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

    plt.savefig((Case_loc+'NoBadpoints_%s_%d_Clusters_SilhouetteScore_Distribution'+'.png')%(RemoveBadSamples, n_clusters))

#def fig_AvgSilhouetteScore_Nclusters(Case_loc, RemoveBadSamples, range_n_clusters, list_silhouette_avg, n_components):
#    #pdb.set_trace()
#    fig = plt.figure(figsize=(15, 8))
#    ax = fig.add_subplot(111)
#    ax.plot(np.asarray(range_n_clusters), np.asarray(list_silhouette_avg), marker='o')
#    plt.title(('NoBadpoints_%s_AvgSilhouetteScore_NClusters')%(RemoveBadSamples))
#    plt.savefig((Case_loc+'NoBadpoints_%s_AvgSilhouetteScore_NClusters_%dD'+'.png')%(RemoveBadSamples, n_components))

def fig_PerfMetric_Nclusters(Case_loc, RemoveBadSamples, range_n_clusters, PerfMetric, PerfMetricName, n_components):
    #pdb.set_trace()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(np.asarray(range_n_clusters), np.asarray(PerfMetric), marker='o')
    plt.title(('NoBadpoints_%s_%s_NClusters')%(RemoveBadSamples, PerfMetricName))
    plt.savefig((Case_loc+'NoBadpoints_%s_%s_NClusters_%dD'+'.png')%(RemoveBadSamples, PerfMetricName, n_components))

#def fig_heatmap(Data_matrix, n_clusters, cluster_labels):
    #get similarity matrix
    #sorting issues?
    #convert to seaborn DataFrame
    #draw figures
    
def export_vector_values(vec1_str, vec1, vec2_str, vec2,
                         Case_loc):
    #pdb.set_trace()
    out_file_loc = ((Case_loc+'%s_and_%s'+'.txt')%(vec1_str, vec2_str))
    out_file = open(out_file_loc, 'w')

    print('export %s and %s to %s'%(vec1_str, vec2_str, out_file_loc))

    for i in range(len(vec1)):
        line = '%g \t %g'%(vec1[i], vec2[i])
        out_file.write(line)
        out_file.write('\n')

    out_file.close()

    print('\texport %s and %s to %s'%(vec1_str, vec2_str, out_file_loc))


def export_samples_properties(Samples_Properties, Case_loc, dm_method, n_components, n_clusters):
    #pdb.set_trace()

    out_file_loc = ((Case_loc+'NoBadpoints_%s_n_clusters_%d_%dD'+'.txt')%(RemoveBadSamples, n_clusters, n_components))
    out_file = open(out_file_loc, 'w')

    print('export sample properties to %s'%out_file_loc)

    line = 'Sample_label' + '\tIs_Good_Sample' + '\tCluster_Label' + '\tSilhouette_Score'
    out_file.write(line)
    out_file.write('\n')
    
    for i in range(len(Samples_Properties)):
        line = ('%s' + '\t%d' + '\t%d' + '\t%g')%(Samples_Properties[i][0], #0: Sample label
                                                  Samples_Properties[i][1], #1: Good/Bad sample
                                                  Samples_Properties[i][2], #2: Kmeans-cluster-label
                                                  Samples_Properties[i][3])  #3: silhouette score
        out_file.write(line)
        out_file.write('\n')

    out_file.close()

    print('\texport sample properties to %s ends'%out_file_loc)
    

## Data_matrix Sample_labels GoodSamples_index Pdist Case_loc dm_method n_components
def process_n_clusters(Data_matrix, Sample_labels, GoodSamples_index, BadSamples_index,
                       n_clusters, dm_method, Pdist, n_components,
                       Case_loc):

    print('\nprocess n_clusters=%d' % n_clusters)
    #pdb.set_trace()

    Data_matrix_orig = Data_matrix

    if reduce_data_dim == True:
    # ---------- Reduce Data Dim before doing clustering
        if reduce_data_dim_method == 'tsne':
            #pdb.set_trace()
            Data_matrix = dm_tsne(Data_matrix, reduce_data_dim_n_components)
        elif reduce_data_dim_method == 'isomap':
            Data_matrix = dm_isomap(Data_matrix, reduce_data_dim_n_neighbors, reduce_data_dim_n_components)
        elif reduce_data_dim_method == 'lle':
            pdb.set_trace()
            Data_matrix = dm_lle(Data_matrix, reduce_data_dim_n_neighbors, reduce_data_dim_n_components, reduce_data_dim_lle_method)
        elif reduce_data_dim_method == 'pca':
            #pdb.set_trace()
            Data_matrix = dm_pca(Data_matrix, reduce_data_dim_n_components)
        elif reduce_data_dim_method == 'sparse-pca':
            pdb.set_trace()
        else:
            pdb.set_trace()
            print('undefined dm method')

    # ---------- Clustering
    if clustering_method == 'kmeans':
        #kmeans - to get colors for different clusters
        [cluster_labels, color_matrix, SD] = clst_kmeans(Data_matrix, n_clusters)
    elif clustering_method == 'hier':
        #pdb.set_trace()
        [cluster_labels, color_matrix, SD] = clst_hier(Data_matrix, Linkage_Method, Pdist, n_clusters)
        #? SD
    elif clustering_method == 'kmedoid':
        #TBD
        #? SD
        pdb.set_trace()
        [cluster_labels, color_matrix] = clst_kmedoid(Data_matrix, n_clusters)
    else:
        pdb.set_trace()
        print('unknown clustering_method')

    #pdb.set_trace()
    [sample_silhouette_values, silhouette_avg] = get_silhouette_info(Data_matrix_orig, cluster_labels, Pdist)
    #if np.isnan(silhouette_avg):
    #pdb.set_trace()
    #draw silhouette_distribution for certain n_clusters
    #pdb.set_trace()

    #heat map
    #pdb.set_trace()
    if exportHeatmap==True:
        draw_heatmap.fig_heatmap(Data_matrix_orig, cluster_labels, sample_silhouette_values, #draw_heatmap2.fig_heatmap2
                                 Sample_labels, GoodSamples_index, 
                                 Pdist,
                                 Case_loc, RemoveBadSamples, n_clusters)

    fig_silhouette_distribution(Data_matrix_orig,
                                sample_silhouette_values, silhouette_avg,
                                n_clusters, cluster_labels,
                                Case_loc)
    
    # ---------- Dimension Reduction
    #samples' properties
    Samples_Properties = list()
    #Samples[i] = [Sample label, Good/Bad sample, Kmeans-cluster-label, silhouette score]
    #pdb.set_trace()
    for i in range(len(Data_matrix)):
        ith_list = list()
        #0: Sample label
        ith_list.append(Sample_labels[i])
        #1: Good/Bad sample
        if i in GoodSamples_index:
            ith_list.append(1)
        else:
            ith_list.append(0)
        #2: Kmeans-cluster-label
        ith_list.append(cluster_labels[i])
        #3: silhouette score
        #pdb.set_trace()
        ith_list.append(sample_silhouette_values[i])
        Samples_Properties.append(ith_list)

    #try to print sample properties
    export_samples_properties(Samples_Properties, Case_loc, dm_method, n_components, n_clusters)

    #pdb.set_trace()

    NonNegativeSilhouette_Index = [i for i in range(len(Samples_Properties)) if Samples_Properties[i][3]>=0]
    NegativeSilhouette_Index = [i for i in range(len(Samples_Properties)) if Samples_Properties[i][3]<0]

    #dimension reduction
    #isomap
    if dm_method == 'tsne':
        #pdb.set_trace()
        Y = dm_tsne(Data_matrix, n_components)
    elif dm_method == 'isomap':
        Y = dm_isomap(Data_matrix, n_neighbors, n_components)
    elif dm_method == 'lle':
        pdb.set_trace()
        Y = dm_lle(Data_matrix, n_neighbors, n_components, lle_method)
    elif dm_method == 'pca':
        #pdb.set_trace()
        Y = dm_pca(Data_matrix, n_components)
    elif dm_method == 'sparse-pca':
        Y = dm_sparse_pca(Data_matrix, n_components, alpha, ridge_alpha)
    else:
        pdb.set_trace()
        print('undefined dm method')
        return silhouette_avg

    #visualization
    fig_scatter(Y, color_matrix, dm_method, n_components,
                Case_loc, n_clusters, RemoveBadSamples,
                GoodSamples_index, BadSamples_index, NonNegativeSilhouette_Index, NegativeSilhouette_Index)
    print('\n')
    return [silhouette_avg, SD] #SD: sum of distortion

# ---------- main procedure
# read data & preprocessing
[RNA_labels, Sample_labels, Data_matrix] = load_data(Data_loc, read_mode)
#pdb.set_trace()
Data_matrix = np.asarray(Data_matrix, dtype=np.float64) #list to array
Data_matrix = Data_matrix.transpose() # N (samples) by M (features), row_labels=Sample_labels; 

#cut off (e.g. RNA 6~7 reads: ignore)
#N/A
#check histogram of reads per sample to decide cutoff thresholds
#pdb.set_trace()
#check_histogram_of_sample_reads(Data_matrix, bins)

#pdb.set_trace()

if RemoveBadSamples == True:
    Data_matrix = [row for row in Data_matrix if sum(row)>BadSampleThreshold]

[GoodSamples_index, BadSamples_index] = FilterBadSamples(Data_matrix, BadSampleThreshold)

#normalization
for i in range(len(Data_matrix)):
    if sum(Data_matrix[i])!=0: # to avoid 'nan'
        #pdb.set_trace()
        Data_matrix[i] = Data_matrix[i]/sum(Data_matrix[i])#*Data_matrix[i])

#for a fixed n_cluster
list_silhouette_avg = list()
list_SD = list()
for n_clusters in range_n_clusters:    
    Case_loc_sub = (Case_loc + 'RemoveBadSamples_%s/%d_Clusters/')%(RemoveBadSamples,n_clusters)
    if not os.path.exists(Case_loc_sub):
        os.makedirs(Case_loc_sub)

    [silhouette_avg, SD] = process_n_clusters(Data_matrix, Sample_labels, GoodSamples_index, BadSamples_index,
                                        n_clusters, dm_method, Pdist, n_components,
                                        Case_loc_sub)
    list_silhouette_avg.append(silhouette_avg)
    list_SD.append(SD)

#fig_AvgSilhouetteScore_Nclusters(Case_loc, RemoveBadSamples, range_n_clusters, list_silhouette_avg, n_components)
#pdb.set_trace()
fig_PerfMetric_Nclusters(Case_loc, RemoveBadSamples, range_n_clusters, list_silhouette_avg, 'AvgSilhouette', n_components)
#pdb.set_trace()
export_vector_values('n_clusters', range_n_clusters, 'avg_silhouette', list_silhouette_avg, Case_loc)
fig_PerfMetric_Nclusters(Case_loc, RemoveBadSamples, range_n_clusters, list_SD, 'SumOfDistortion', n_components)


