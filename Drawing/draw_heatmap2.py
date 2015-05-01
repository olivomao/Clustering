import matplotlib.pyplot as plt
import matplotlib.figure as fg
import scipy.spatial.distance as dt
#import pandas as pd
#import seaborn as sns
import numpy as np
import pdb

#try to draw heatmap not using seaborn & pandas

def fig_heatmap2(data_matrix, data_labels,
                 Sample_labels, GoodSamples_index, 
                 Pdist,
                 Case_loc, RemoveBadSamples, n_clusters):
    #pdb.set_trace()

    # prepare indexes
    if len(data_matrix)!=len(Sample_labels): #if there are bad samples removed
        Sample_labels_2 = [ Sample_labels[i] for i in GoodSamples_index ]
    else: #if bad samples are kept
        Sample_labels_2 = Sample_labels
    # add cluster label to data sample
    Sample_labels_2 = [ ((Sample_labels_2[i]+'(%d)')%data_labels[i]) for i in np.arange(len(Sample_labels_2))]
    Sample_labels_2 = np.asarray(Sample_labels_2) # array
    
    data_matrix = np.asarray(data_matrix, dtype=np.float64) # array
    
    data_matrix_clustered = list() #order samples according to cluster labels
    sample_labels_clustered = list()
    for ith_cluster in range(n_clusters):
        ith_cluster_data = data_matrix[data_labels==ith_cluster]
        data_matrix_clustered = data_matrix_clustered + [row for row in ith_cluster_data]
        ith_cluster_labels = Sample_labels_2[data_labels==ith_cluster]
        sample_labels_clustered = sample_labels_clustered + [row for row in ith_cluster_labels]
    
    pdb.set_trace()
    #gen heat matrix (can be distance or correlation matrix)
    heat_matrix = dt.pdist(data_matrix, Pdist)
    heat_matrix = dt.squareform(heat_matrix)

    # to draw heatmap
    fig, ax = plt.subplots()
    ax.pcolor(heat_matrix, cmap=plt.cm.Reds, edgecolors='k')
    ax.set_xticks(np.arange(0, len(heat_matrix))+0.5)
    ax.set_yticks(np.arange(0, len(heat_matrix))+0.5)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(sample_labels_clustered, minor=False, fontsize=10)
    ax.set_yticklabels(sample_labels_clustered, minor=False, fontsize=10)

    plt.show()

    pdb.set_trace()

########## draft code
    print('fig_heatmap')
        
    #convert to seaborn DataFrame & draw
    #data_matrix_2 = np.asarray(data_matrix, dtype=np.float64).transpose()
    #df = pd.DataFrame(data_matrix_2, columns=Sample_labels_2)
    data_matrix_2 = np.asarray(data_matrix_clustered, dtype=np.float64).transpose()
    
    
