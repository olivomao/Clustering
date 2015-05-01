import pdb
import scipy.spatial.distance as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure as fg
import numpy as np

def fig_heatmap(data_matrix, data_labels, sample_silhouette_values,
                Sample_labels, GoodSamples_index, 
                Pdist,
                Case_loc, RemoveBadSamples, n_clusters):

    #pdb.set_trace()
    if len(data_matrix)!=len(Sample_labels): #bad samples removed
        #pdb.set_trace()
        Sample_labels_2 = [ Sample_labels[i] for i in GoodSamples_index ]
    else: #bad samples kept
        pdb.set_trace()
        Sample_labels_2 = Sample_labels
    #pdb.set_trace()
    Sample_labels_2 = [ (('(%d)'+Sample_labels_2[i]+'(%.4g)')%(data_labels[i], sample_silhouette_values[i])) for i in np.arange(len(Sample_labels_2))] # format: (ith-cluster) sample name (sil val)
    ##pdb.set_trace()
    
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    Sample_labels_2 = np.asarray(Sample_labels_2)

    sil = [[i, sample_silhouette_values[i]] for i in range(len(sample_silhouette_values))]
    sil = np.asarray(sil)
    ##pdb.set_trace()

    data_matrix_clustered = list()
    sample_labels_clustered = list()
    for ith_cluster in range(n_clusters):
        #sort data samples in same cluster according to their silhouette values
        ##pdb.set_trace()

        ith_cluster_sil = -sil[data_labels==ith_cluster] #sil sorted in descending order
        ith_cluster_sil = ith_cluster_sil[np.argsort(ith_cluster_sil[:,1])]
        ##pdb.set_trace()

        ith_cluster_data = [data_matrix[-ith_cluster_sil[i,0]] for i in range(len(ith_cluster_sil))]
        ##pdb.set_trace()

        #ith_cluster_data = data_matrix[data_labels==ith_cluster]
        #[ data_matrix[i] for i in np.arange(len(data_matrix)) and data_labels[i]==ith_cluster ]
        data_matrix_clustered = data_matrix_clustered + [row for row in ith_cluster_data]
        ##pdb.set_trace()

        ith_cluster_labels = [Sample_labels_2[-ith_cluster_sil[i,0]] for i in range(len(ith_cluster_sil))]
        ##pdb.set_trace()

        #ith_cluster_labels = Sample_labels_2[data_labels==ith_cluster]
        #[ Sample_labels_2[i] for i in np.arange(len(Sample_labels_2)) and data_labels[i]==ith_cluster ]
        sample_labels_clustered = sample_labels_clustered + [row for row in ith_cluster_labels]
        ##pdb.set_trace()
    
    #pdb.set_trace()
    print('fig_heatmap')

    #gen sim matrix
    #Dist_matrix = dt.pdist(data_matrix, Pdist)
    #Dist_matrix = dt.squareform(Dist_matrix)

    #convert to seaborn DataFrame & draw
    #data_matrix_2 = np.asarray(data_matrix, dtype=np.float64).transpose()
    #df = pd.DataFrame(data_matrix_2, columns=Sample_labels_2)
    data_matrix_2 = np.asarray(data_matrix_clustered, dtype=np.float64).transpose()
    df = pd.DataFrame(data_matrix_2, columns=sample_labels_clustered)
    corrmat = df.corr()

    #pdb.set_trace()

    f, ax = plt.subplots(figsize=(12,12))
    #f = fg.Figure(figsize=(18,18))
    sns.heatmap(corrmat, vmax=.8, linewidths=0, square=True)

    #pdb.set_trace()
    cols = corrmat.columns.get_level_values(0)
    for i, col in enumerate(cols):
        if i and col != cols[i-1]:
            ax.axhline(len(cols)-i, c='w')
            ax.axvline(i, c='w')
    f.tight_layout()
    f.savefig((Case_loc+\
               'NoBadpoints_%s_%d_Clusters_heatmap'+'.png')\
              %(RemoveBadSamples, n_clusters))
    f.clf()

    #pdb.set_trace()
    
    
    return
