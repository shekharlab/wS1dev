#import
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scanpy as sc
import pandas as pd
from harmony import harmonize
from sklearn.metrics.cluster import adjusted_rand_score
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.utils import shuffle
from anndata import AnnData
from sklearn.cluster import KMeans
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable
from scipy.stats import hypergeom
import sklearn.preprocessing
import seaborn as sn
from random import sample
import pickle
#from matplotlib_venn import venn3, venn3_circles
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


from matplotlib import gridspec
from sklearn.metrics import confusion_matrix, adjusted_rand_score, roc_curve, auc, classification_report, f1_score, cohen_kappa_score
import plotly.graph_objects as go
from itertools import cycle, islice
from sklearn.preprocessing import label_binarize

import scanpy as sc
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.set_figure_params(dpi=100, dpi_save=200)


import scrublet as scr

from typing import Optional

# import warnings
# warnings.filterwarnings("ignore")

import matplotlib as mpl
from matplotlib import gridspec
import xgboost as xgb
from sklearn.metrics import accuracy_score


sc.settings.verbosity = 0           # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.set_figure_params(dpi=75, dpi_save=200)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def test_func():
    print('test')
    
def test2():
    print('test2')
def rem_fem(adata):
    for j in list(adata.obs.Type_nn_dists.values.categories):
        if ('Fem' in j):
            adata = adata[adata.obs['Type_nn_dists']!=j,:]
            adata = adata[adata.obs['Type_leiden']!=j,:]
            adata = adata[adata.obs['Type']!=j,:]
    return adata
def pipeline_short(adata, batch_correct, batch_ID):
                   
    if sp.sparse.issparse(adata.X):
        if np.any(adata.X.A<0):
            raise Exception("Matrix contains negative values")
    else: 
        if np.any(adata.X<0):
            raise Exception("Matrix contains negative values")

    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_ID) #HVGs
    adata.raw = adata
    sc.pp.scale(adata, max_value=10) #scale
    sc.tl.pca(adata, svd_solver='arpack') #run PCA
    
    if (batch_correct):
        Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = batch_ID)
        adata.obsm['X_harmony'] = Z
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_harmony', n_pcs=40)
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

    else:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
        sc.tl.leiden(adata)
        sc.tl.umap(adata)
        
def pre_process(adata, batch_ID):
    
    if sp.sparse.issparse(adata.X):
        if np.any(adata.X.A<0):
            raise Exception("Matrix contains negative values")
    else: 
        if np.any(adata.X<0):
            raise Exception("Matrix contains negative values")
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_ID) #HVGs
    adata.raw = adata
    sc.pp.scale(adata, max_value=10) #scale
    sc.tl.pca(adata, svd_solver='arpack') #run PCA
    

def pipeline(adata, 
             batch_correct: Optional[bool] = None,
             batch_ID: Optional[str] = None):
    
    if (batch_correct):
        Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = batch_ID)
        adata.obsm['X_harmony'] = Z
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_harmony', n_pcs=40)
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

    else:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
        sc.tl.leiden(adata)
        sc.tl.umap(adata)

def clust_obs_plot(adata, obs_id, obs_id_split):

    obs_quants = []
    for i in adata.obs[obs_id_split].values.categories:
    #for i in dr_corr.obs['Atlas Label'].values.categories:
        obs_quants.append(np.mean(adata[adata.obs[obs_id_split]==i].obs[obs_id].values))
        
    x, y = adata.obs[obs_id_split].values.categories, obs_quants
    ser = pd.Series(y,x).sort_values(ascending=False)
    
    x,y = ser.index, ser.values
    
    plt.figure(figsize=(15,4))
    plt.bar(x,y)
    plt.axhline(np.median(adata.obs[obs_id]), color='r')
    plt.ylabel(obs_id)
    
    return x, y

def DE(adata, obs_id, obs_id_test, ref, pts_thresh, lf_thresh):

    sc.tl.rank_genes_groups(adata, groupby=obs_id, groups=[obs_id_test], 
                                reference=ref, method='t-test', pts=True, use_raw=True)

    lfcs = adata.uns['rank_genes_groups']['logfoldchanges'].astype([(obs_id_test, '<f8')]).view('<f8') 

    l231_genes = adata.uns['rank_genes_groups']['pts']

    lfcs = []
    p_adj = []
    names = list(adata.uns['rank_genes_groups']['names'].astype([(obs_id_test, '<U50')]).view('<U50'))
    logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges'].astype([(obs_id_test, '<f8')]).view('<f8')
    pvals_adj = adata.uns['rank_genes_groups']['pvals_adj'].astype([(obs_id_test, '<f8')]).view('<f8')

    for i in l231_genes.index:
        lfcs.append(logfoldchanges[names.index(i)])
        p_adj.append(pvals_adj[names.index(i)])

    l231_genes['LF'] = lfcs
    l231_genes['p_adj'] = p_adj
    
    #plt.hist(l231_genes[obs_id_test].values)

    l231_genes = l231_genes[l231_genes[obs_id_test]>pts_thresh]
    
    sort_LF = l231_genes.sort_values('LF', ascending=False)
    
    a = np.where(sort_LF[sort_LF['LF']>0.6]['p_adj'].values<0.05)[0].shape[0]
    b = sort_LF[sort_LF['LF']>0.6].shape[0]
    if(a == b):
        print('cutoffs are good at 1.5 FC level')
    else:
        print(a,b)
        
    return sort_LF[sort_LF['LF']>lf_thresh]
        
def clust_obs(adata, obs_id, obs_id_split):

    obs_quants = []
    for i in adata.obs[obs_id_split].values.categories:
    #for i in dr_corr.obs['Atlas Label'].values.categories:
        obs_quants.append(np.mean(adata[adata.obs[obs_id_split]==i].obs[obs_id].values))
        
    x, y = adata.obs[obs_id_split].values.categories, obs_quants
    ser = pd.Series(y,x).sort_values(ascending=False)
    return ser.index, ser.values

def res_tune(adata, def_labels):
    
    adata_ari = []
    adata_num_clusts = []
    for res in np.arange(1,2.1, 0.1):
        sc.tl.leiden(adata, resolution=res)

        adata_num_clusts.append(len(adata.obs.leiden.values.categories))
        adata_ari.append(adjusted_rand_score(def_labels, list(adata.obs.leiden.values)))
        
    return adata_num_clusts, adata_ari

from math import log, e
def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

def check_clust(adata, clus, ref_age):
    sc.pl.umap(adata[adata.obs.leiden==clus], color=[ref_age+'-2022 Mapping Prob', ref_age+'-2022 Mapping Label'])
    print(adata[adata.obs.leiden==clus].shape, 'mean prob', np.mean(adata[adata.obs.leiden==clus].obs[ref_age+'-2022 Mapping Prob']))
    ser = adata[adata.obs.leiden==clus].obs['Sample'].value_counts()
    plt.bar(ser.index, ser.values,)
    plt.xticks(rotation=90)
    plt.show()
    ser = adata[adata.obs.leiden==clus].obs[ref_age+'-2022 Mapping Label'].value_counts()
    plt.bar(ser.index, ser.values,)
    plt.xticks(rotation=90)
    plt.show()
    
def maj_vote_annot(adata, clust_id, putative_annot, final_annot, vlmcendo=False):
    clust_list = []
    for i in adata.obs[clust_id].values.categories:
        clust = adata[adata.obs[clust_id]==i,:]
        clust_df = clust.obs[putative_annot].value_counts()
        if (clust_df.index[0]=='Unassigned'):
            biggest_cat = clust_df.index[1]
        else: biggest_cat = clust_df.index[0]
        if (vlmcendo):
            if (biggest_cat in ('VLMC', 'Endo')):
                clust.obs[putative_annot+'_maj'] = ['VLMC+Endo']*clust.shape[0]
                clust_list.append(clust)
            else:
                clust.obs[putative_annot+'_maj'] = [biggest_cat]*clust.shape[0]
                clust_list.append(clust)
        else:
            clust.obs[putative_annot+'_maj'] = [biggest_cat]*clust.shape[0]
            clust_list.append(clust)

    adata_annot = clust_list[0].concatenate(clust_list[1:], index_unique=None)
    adata_annot.obs[final_annot] = adata_annot.obs[putative_annot+'_maj']
    a = sc.pl.dotplot(adata_annot, 'Mdga1', 'leiden',) #to fix categories
    return adata_annot

def clus_sample_bars(adata, a_, b_, samp_id, clus_id, size=(13,9.5)):
    adata_ser = adata.obs[samp_id].value_counts(normalize=True)
    plt.bar(adata_ser.index, adata_ser.values)
    plt.xticks(rotation=90)
    plt.title('Overall Dataset')
    plt.show()
    a,b = 0 ,0 
    fig, axs = plt.subplots(a_,b_, figsize=size)

    for i in adata.obs[clus_id].values.categories:
        clus = adata[adata.obs[clus_id]==i,:]
        clus_ser = clus.obs[samp_id].value_counts(normalize=True)
        axs[a,b].bar(clus_ser.index, clus_ser.values)
        axs[a,b].set_title(i+' ('+str(clus.shape[0])+' cells)')
        axs[a,b].set_ylabel('Fraction')
        axs[a,b].set_xticklabels(clus_ser.index, rotation=90)
        b = b + 1
        if (b>=b_):
            b = 0
            a = a + 1
    plt.tight_layout()
    
    
def euclidean_distance(vector1, vectors_list):
    # Ensure that the input vectors are numpy arrays
    vector1 = np.array(vector1)
    vectors_list = [np.array(vector) for vector in vectors_list]
    
    # Calculate the Euclidean distance between the input vector and each vector in the list
    distances = [np.linalg.norm(vector1 - vector) for vector in vectors_list]
    return distances

def nn_voting(adata, type_old, type_new, delta_thresh):
    adata.obs['idx'] = np.arange(adata.shape[0])

#     #re-make graph using UMAP space for voting below
#     sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_umap')
#     sc.tl.umap(adata)

    adata.obs[type_new] = adata.obs[type_old]
    adata_un_idx = adata[adata.obs[type_old]=='Unassigned',:].obs.idx.values #indices of unassigned cells
    neigh_matx = adata.obsp['distances'].A #each row's nonzero entries tells neighbors
    adata_iGBs = list(adata.obs[type_old].values) #all the step1 assignments
    n_neighbs = adata.uns['neighbors']['params']['n_neighbors']
    print(adata.uns['neighbors']['params'])
    print(' ')
    
    print("Pre-voting unassigned: ", adata_iGBs.count('Unassigned'),
          adata_iGBs.count('Unassigned')/adata.shape[0])
    
    
    delta = 1 #init delta
    while (delta>delta_thresh):    
        un_frac1 = adata_iGBs.count('Unassigned')/adata.shape[0] #frac of unassigned cells before voting

        #loop thru each cell
        for i in adata_un_idx:

            #so that it only loops thru still-unassigned cells after first pass
            if (adata.obs[type_new][i]=='Unassigned'):
                neighbs_idx = np.where(neigh_matx[i,:]>0)[0] #cell i's neighbors
                neighbs_iGBs = adata[neighbs_idx].obs[type_new] #the neighbors' iGBs

#                 #if there's a type in the neighbors that is majority, assign it
#                 if (neighbs_iGBs.value_counts()[0] > n_neighbs/2):
#                     adata_iGBs[i] = neighbs_iGBs.value_counts().index[0]
                
                #no need to be majority, just be biggest one
                adata_iGBs[i] = neighbs_iGBs.value_counts().index[0]
                #update the IDs to help assignment of next cell
                adata.obs[type_new] =  pd.Categorical(adata_iGBs)

        un_frac2 = adata_iGBs.count('Unassigned')/adata.shape[0] #frac of unassigned cells after voting

        delta = (un_frac1-un_frac2)/un_frac1   #stop when this changes by less than 1%
        print(delta, un_frac2)

    print("Post-voting unassigned: ", adata_iGBs.count('Unassigned'),
          adata_iGBs.count('Unassigned')/adata.shape[0])
    
    return adata


def make_hmap(adata,thres, y_obs='Type', x_obs='leiden', ):
    crosstab_data = pd.crosstab(adata.obs[y_obs], adata.obs[x_obs], 
                           normalize='index')
    # For rows
    row_order = np.argsort(-crosstab_data.values.max(axis=1))

    # For columns
    column_order = np.argsort(-crosstab_data.values.max(axis=0))

    reordered_rows = crosstab_data.index[row_order]
    reordered_crosstab_rows = crosstab_data.iloc[row_order]

    # For columns
    reordered_columns = crosstab_data.columns[column_order]
    reordered_crosstab = reordered_crosstab_rows[reordered_columns]
    #return reordered_crosstab
    
    
    from sklearn.metrics import adjusted_rand_score

    

    labels_true = adata.obs[y_obs]
    labels_pred = adata.obs[x_obs]

    # Compute the Adjusted Rand Index
    ari = adjusted_rand_score(labels_true, labels_pred)
    
    return [reordered_crosstab.applymap(lambda value: value if value > thres else np.round(value, 2)),
           ari]
def freq_scatter(x, y, x_lab, y_lab, hue_, unity_lim, low_lim, hue_ord=None):

    if (x.shape[0]!=y.shape[0]): 
        print('x and y are different by', np.abs(x.shape[0]-y.shape[0]), 'categories')
    
    if (y.shape[0]>x.shape[0]):
        y = y[x.index]
    else:
        x = x[y.index]

    df = pd.DataFrame(index=list(x.index.values), columns=[x_lab, y_lab],
                      data=np.transpose(np.array([x.values, y.values])))

    df[hue_] = df.index
    sn.scatterplot(data=df, x=x_lab, y = y_lab, hue=hue_, s=50, hue_order=hue_ord)
    plt.legend(bbox_to_anchor=(1.01, 1.03), loc='upper left', fontsize=14, ncol=2)
    plt.plot(np.linspace(low_lim,unity_lim), np.linspace(low_lim,unity_lim), ls='--', 
             color='black', linewidth=0.75)
    plt.title('Pearson R: '+str(np.round(sp.stats.pearsonr(x, y)[0], 3)))
    plt.grid(False)
    #plt.loglog()
    


from anndata import AnnData
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable

def module_score(adata:AnnData, genes_use: list, score_name: Optional[str] = None, verbose: bool = True):
    
    """\
    Compute module scores for all cells in adata as described in methods of RGC-dev paper.
    
    
    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    genes_use
        list of genes in module of interest
    score_name
        Name endowed to the module score to be computed
        e.g. "Mod1"
    verbose
        Inform user of fraction of module genes that are in adata
        
    Returns
    -------
    adata with a new .obs called score_name
    
    """
    
    if (score_name==None):
        score_name = str(input("Provide a name for this score (no spaces): "))
        
    genes_use0 = genes_use
    genes_use = list(set(genes_use).intersection(adata.var_names))#genes that are both in module and `adata`
    
     
    if (len(genes_use) == 0):
        raise ValueError("Error : Must provide a list of genes that are present in the data")
        
    
    if (verbose):
        if(len(genes_use0) > len(genes_use)):
            n = len(genes_use0) - len(genes_use)
            print(score_name,": Note that", n, "of the", len(genes_use0), "genes in your module do not exist in the data set." )
        else:
            print(score_name,": Note that all of the", len(genes_use), "genes in your module are in the data set." )
    
    
    
    adata_score = adata.copy()
    adata_score = adata[:,genes_use]
    
    counts_modgenes = adata_score.X.toarray() #all cells, module genes
    counts_all = adata.X.toarray() #all cells, all genes
    #scores = np.mean(counts_modgenes, axis=1) - np.mean(counts_all, axis=1) #(row means of counts_modgenes ) - (row means of counts_all)
    scores = np.mean(counts_modgenes, axis=1) #(row means of counts_modgenes )

    adata.obs[score_name] = scores
    
    return genes_use    
    

def clean_subclasses(adata, rep):
    #define avg PC position of each type. Gotta use a type with no "Unassigned" group
    typ_mean_dict = {}
    for i in adata.obs['Subclass'].values.categories:
        typ_mean_dict[i] = np.mean(adata[adata.obs['Subclass']==i,:].obsm[rep][:,0:40], axis=0)

    #assign based on proximity to avg type
    dists_list = []
    dict_keys = list(typ_mean_dict.keys())
    for i in range(adata.shape[0]):
        typ_ = adata.obs['Subclass'][i]
        dists = euclidean_distance(adata.obsm[rep][i,0:40], list(typ_mean_dict.values()))

        #dist = dists[dict_keys.index(typ_)]
        typ = dict_keys[np.argmin(dists)]
        #types.append(typ)
        if (typ!=typ_):
            dists_list.append(100)
        else: dists_list.append(0)

    adata.obs['Dist to Subclass'] = dists_list
    
def make_dict(adata, obs_id):

    adata_dict = {}
    for num,i in enumerate(adata.obs[obs_id].values.categories):
        adata_dict[i] = num
        
    #adata_dict['Unassigned'] = num + 1
    
    return adata_dict
    

def plotConfusionMatrix(
    ytrue,
    ypred,
    type,
    xaxislabel,
    yaxislabel,
    title,
    train_dict,
    test_dict=None,
    re_order=None,
    re_order_cols = None,
    re_index = None,
    re_order_rows = None,
    save_as=None,):
    
    #very bad
    numbertrainclasses = len(set(ypred))
    numbertestclasses = len(set(ytrue))
    
    #cfm is 11x11 b/c 11 is = y_true U y_pred
    confusionmatrix = confusion_matrix(y_true = ytrue, y_pred = ypred)
    
    #only need this when mapping b/c if validaiton, all classes will be used and cfm will be constructed properly
    if type == 'mapping':
        rows = np.where(np.sum(confusionmatrix, axis=1)>0)[0]
        cols = np.where(np.sum(confusionmatrix, axis=0)>0)[0]

        cfm = confusionmatrix[rows,:][:,cols]
        
        #show all columns, even ones with no mapping
        #cfm_z = np.zeros((len(test_dict),len(train_dict)))
        cfm_z = np.zeros((len(rows),len(cols)))

        
        
        cfm_z[:, np.array(pd.Categorical(ypred).categories, dtype='int')]=cfm 
        confusionmatrix = cfm_z
        #always keep only as many as rows as num of test classes
        #but, b/c of python's 0 indexing, if the number of training classes is in the 
        #y_pred list, then that means there was an unassigned
#       if numbertrainclasses in ypred:
#         confusionmatrix = confusionmatrix[0:numbertestclasses,0:numbertrainclasses+1]#for Unassigned
#       else:
#         confusionmatrix = confusionmatrix[0:numbertestclasses,0:numbertrainclasses]
    
        confmatpercent = confusionmatrix/np.sum(confusionmatrix, axis=1).reshape(-1,1)

        conf_df = pd.DataFrame(confmatpercent)
        conf_df.index = list(test_dict.keys())

        #name columns of conf mat
        if(len(conf_df.columns)>len(train_dict)):
            conf_df.columns = list(train_dict.keys())+['Unassigned']
        else:
            conf_df.columns = list(train_dict.keys())


        if (re_order):
            conf_df = conf_df[re_order_cols]

        if (re_index):
            conf_df = conf_df.reindex(re_order_rows)

        diagcm = conf_df.to_numpy()
    
    
        xticksactual = list(conf_df.columns)
        
        #print(conf_df)
        
    
    else:
        confmatpercent = np.zeros(confusionmatrix.shape)
        for i in range(confusionmatrix.shape[0]):
            if np.sum(confusionmatrix[i,:]) != 0:
                confmatpercent[i,:] = confusionmatrix[i,:]/np.sum(confusionmatrix[i,:])
            else:
                confmatpercent[i,:] = confusionmatrix[i,:]
            diagcm = confmatpercent
            xticks = np.linspace(0, confmatpercent.shape[1]-1, confmatpercent.shape[1], dtype = int)
        xticksactual = []
        for i in xticks:
            if i != numbertrainclasses:
                xticksactual.append(list(train_dict.keys())[i])
            else:
                xticksactual.append('Unassigned')
        
    dot_max = np.max(diagcm.flatten())
    dot_min = 0
    if dot_min != 0 or dot_max != 1:
        frac = np.clip(diagcm, dot_min, dot_max)
        old_range = dot_max - dot_min
        frac = (frac - dot_min) / old_range
    else:
        frac = diagcm
    xvalues = []
    yvalues = []
    sizes = []
    for i in range(diagcm.shape[0]):
        for j in range(diagcm.shape[1]):
            xvalues.append(j)
            yvalues.append(i)
            sizes.append((frac[i,j]*35)**1.5)
    size_legend_width = 0.5
    height = diagcm.shape[0] * 0.3 + 1
    height = max([1.5, height])
    heatmap_width = diagcm.shape[1] * 0.35
    width = (
        heatmap_width
        + size_legend_width
        )
    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.02,
        hspace=0.04,
        width_ratios=[
                    heatmap_width,
                    size_legend_width
                    ],
        height_ratios = [0.5, 10]
        )
    dot_ax = fig.add_subplot(axs[1, 0])
    dot_ax.scatter(xvalues,yvalues, s = sizes, c = 'blue', norm=None, edgecolor='none')
    y_ticks = range(diagcm.shape[0])
    dot_ax.set_yticks(y_ticks)
    if type == 'validation':
        dot_ax.set_yticklabels(list(train_dict.keys()))
    elif type == 'mapping':
      #dot_ax.set_yticklabels(list(test_dict.keys()))
        dot_ax.set_yticklabels(list(conf_df.index))
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    dot_ax.set_xticklabels(xticksactual, rotation=90)
    dot_ax.tick_params(axis='both', labelsize='small')
    dot_ax.grid(True, linewidth = 0.2)
    dot_ax.set_axisbelow(True)
    dot_ax.set_xlim(-0.5, diagcm.shape[1] + 0.5)
    ymin, ymax = dot_ax.get_ylim()
    dot_ax.set_ylim(ymax + 0.5, ymin - 0.5)
    dot_ax.set_xlim(-1, diagcm.shape[1])
    dot_ax.set_xlabel(xaxislabel)
    dot_ax.set_ylabel(yaxislabel)
    dot_ax.set_title(title)
    size_legend_height = min(1.75, height)
    wspace = 10.5 / width
    axs3 = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=axs[1, 1],
        wspace=wspace,
        height_ratios=[
                    size_legend_height / height,
                    (height - size_legend_height) / height
                    ]
        )
    diff = dot_max - dot_min
    if 0.3 < diff <= 0.6:
        step = 0.1
    elif diff <= 0.3:
        step = 0.05
    else:
        step = 0.2
    fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        fracs_values = (fracs_legends - dot_min) / old_range
    else:
        fracs_values = fracs_legends
    size = (fracs_values * 35) ** 1.5
    size_legend = fig.add_subplot(axs3[0])
    size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, c = 'blue')
    size_legend.set_yticks(range(len(size)))
    labels = ["{:.0%}".format(x) for x in fracs_legends]
    if dot_max < 1:
        labels[-1] = ">" + labels[-1]
    size_legend.set_yticklabels(labels)
    size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
    size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True)
    size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
    size_legend.spines['right'].set_visible(False)
    size_legend.spines['top'].set_visible(False)
    size_legend.spines['left'].set_visible(False)
    size_legend.spines['bottom'].set_visible(False)
    size_legend.grid(False)
    ymin, ymax = size_legend.get_ylim()
    size_legend.set_ylim(ymin, ymax + 0.5)
    
    if (save_as is not None):
        fig.savefig(save_as, bbox_inches = 'tight')

    return diagcm, xticksactual, axs



#This helper method plots validation plots in sequential order (i.e. first plot is for first batch, second plot is for second batch, etc.)
def plot_validation_plots(validation_label_train_70, valid_predlabels_train_70, train_dict, save_as=None):
    
    ARI = adjusted_rand_score(labels_true = validation_label_train_70, 
                              labels_pred = valid_predlabels_train_70)
    
    
    c = 0
    for i in range(validation_label_train_70.shape[0]):
        if (validation_label_train_70[i]!=valid_predlabels_train_70[i]):
            c = c +1
    acc = (1 - c/len(validation_label_train_70))*100
    
           
    validationconfmat, validationxticks, validationplot = plotConfusionMatrix(
    ytrue = validation_label_train_70,
    ypred = valid_predlabels_train_70,
    train_dict=train_dict,
    type = 'validation',
    save_as = save_as,
    title = 'ARI = {:.3f}, Accuracy = {:.3f}'.format(ARI, acc),
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )

def plot_mapping(test_labels, test_predlabels, test_dict, train_dict, 
                 xaxislabel, yaxislabel,
                re_order=None,
    re_order_cols = None,
                 re_index = None,
    re_order_rows = None, save_as=None):
    
    ARI = adjusted_rand_score(labels_true = test_labels, 
                              labels_pred = test_predlabels)
    NCE = calculateNCE(labels_true = test_labels, labels_pred = test_predlabels)
    
 
    
           
    mappingconfmat, mappingxticks, mappingplot = plotConfusionMatrix(
    ytrue = test_labels,
    ypred = test_predlabels,
    test_dict=test_dict,
    train_dict=train_dict,
    type = 'mapping',
    save_as = save_as,
    title = 'ARI = {:.3f}, NCE = {:.3f}'.format(ARI, NCE),
    xaxislabel =xaxislabel,
    yaxislabel = yaxislabel,
        re_order=re_order,
    re_order_cols = re_order_cols,
        re_index = re_index,
    re_order_rows = re_order_rows,
    ) 
    return mappingconfmat, mappingxticks, mappingplot
      
      
#This helper method uses xgboost to train classifiers.
def trainclassifier(train_anndata, common_top_genes, obs_id, train_dict, eta,
                    max_cells_per_ident, train_frac, min_cells_per_ident):
    
    if sp.sparse.issparse(train_anndata.X):
        if np.any(train_anndata.X.A<0):
            raise Exception("Matrix contains negative values")
    else: 
        if np.any(train_anndata.X<0):
            raise Exception("Matrix contains negative values")

    
    start_time = time.time()
    
    numbertrainclasses = len(train_anndata.obs[obs_id].values.categories)

    xgb_params_train = {
            'objective':'multi:softprob',
            'eval_metric':'mlogloss',
            'num_class':numbertrainclasses,
            'eta':eta,
            'max_depth':4,
            'subsample': 0.6}
    nround = 200
    #Train XGBoost on 70% of training data and validate on the remaining data


    training_set_train_70 = []
    validation_set_train_70 = []
    training_label_train_70 = []
    validation_label_train_70 = []

    #loop thru classes to split for training and validation
    for i in train_anndata.obs[obs_id].values.categories:
        
        #how many cells in a class
        cells_in_clust = train_anndata[train_anndata.obs[obs_id]==i,:].obs_names #cell names
        n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
        
        #sample 70% for training and rest for validation
        train_temp = np.random.choice(cells_in_clust,n,replace = False)
        validation_temp = np.setdiff1d(cells_in_clust, train_temp)
        
        #upsample small clusters
        if len(train_temp) < min_cells_per_ident:
            train_temp_bootstrap = np.random.choice(train_temp, size = min_cells_per_ident - int(len(train_temp)))
            train_temp = np.hstack([train_temp_bootstrap, train_temp])
        
        #store training and validation **names** of cells in vectors, which update for every class
        training_set_train_70 = np.hstack([training_set_train_70,train_temp])
        validation_set_train_70 = np.hstack([validation_set_train_70,validation_temp])
        
        #store training and validation **labels** of cells in vectors, which update for every class
        training_label_train_70 = np.hstack([training_label_train_70,np.repeat(train_dict[i],len(train_temp))])
        validation_label_train_70 = np.hstack([validation_label_train_70,np.repeat(train_dict[i],len(validation_temp))])

        #need train_dict b/c XGboost needs number as class labels, not words
        #this is only deconvulted later in plotting function
        
    #put data in XGB format
    X_train = train_anndata[training_set_train_70,common_top_genes].X
    train_matrix_train_70 = xgb.DMatrix(data = X_train, label = training_label_train_70, 
                                        feature_names = common_top_genes)
    
    X_valid = train_anndata[validation_set_train_70,common_top_genes].X
    validation_matrix_train_70 = xgb.DMatrix(data = X_valid, label = validation_label_train_70, 
                                             feature_names = common_top_genes)

    del training_set_train_70, validation_set_train_70, training_label_train_70
    
    #Train on 70%
    bst_model_train_70 = xgb.train(
        params = xgb_params_train,
        dtrain = train_matrix_train_70,
        num_boost_round = nround)
    
    #Validate on 30%
    #a validation_cells x numclasses matrix, with each vector containing prob association with the classes
    validation_pred_train_70 = bst_model_train_70.predict(data = validation_matrix_train_70)
    
    #for each cell, go through vec of probs and take index of max prob: that's assignment
    valid_predlabels_train_70 = np.zeros((validation_pred_train_70.shape[0]))
    for i in range(validation_pred_train_70.shape[0]):
        valid_predlabels_train_70[i] = np.argmax(validation_pred_train_70[i,:])
        
    
    #Train on 100%
    #Train XGBoost on the full training data
    training_set_train_full = []
    training_label_train_full = []

    for i in train_anndata.obs[obs_id].values.categories.values:
        train_temp = train_anndata.obs.index[train_anndata.obs[obs_id].values == i]
        if len(train_temp) < 100:
            train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
            train_temp = np.hstack([train_temp_bootstrap, train_temp])
        
        #indices of cells in class
        training_set_train_full = np.hstack([training_set_train_full,train_temp])
        
        #labels of cells in class: [label*N_class] stacked onto previous classes
        training_label_train_full = np.hstack([training_label_train_full,np.repeat(train_dict[i],len(train_temp))])


    X_train_full = train_anndata[training_set_train_full,common_top_genes].X
    full_training_data = xgb.DMatrix(data = X_train_full, label = training_label_train_full, 
                                     feature_names = common_top_genes)

    del training_set_train_full, training_label_train_full

    bst_model_full_train = xgb.train(
        params = xgb_params_train,
        dtrain = full_training_data,
        num_boost_round = nround)

    
    
    print('trainclassifier() complete after', np.round(time.time() - start_time), 'seconds')
    
    
    f1 = f1_score(validation_label_train_70, valid_predlabels_train_70, average = None)

    
    #real labels of validation set, predicted labels, classifier.
    #recall these are all integers that are deconvulted later in plotting using the dicts
    return validation_label_train_70, valid_predlabels_train_70, bst_model_full_train, f1


#This helper method predicts the testing cluster labels.
def predict(train_anndata, common_top_genes, bst_model_train_full, test_anndata, 
            train_obs_id, test_dict, test_obs_id):
    
    
    if sp.sparse.issparse(train_anndata.X):
        if np.any(train_anndata.X.A<0):
            raise Exception("Training matrix contains negative values")
    else: 
        if np.any(train_anndata.X<0):
            raise Exception("Training matrix contains negative values")

    if sp.sparse.issparse(test_anndata.X):
        if np.any(test_anndata.X.A<0):
            raise Exception("Testing matrix contains negative values")
    else: 
        if np.any(test_anndata.X<0):
            raise Exception("Testing matrix contains negative values")
  
    
    #Predict the testing cluster labels
    #how many classes mapping to 
    numbertrainclasses = len(train_anndata.obs[train_obs_id].values.categories)
    
    #put testing data into XGB format
    full_testing_data = xgb.DMatrix(data = test_anndata[:,common_top_genes].X, 
                                    feature_names=common_top_genes)
    
    #a testing_cells x numclasses matrix, with each vector containing prob association with the classes
    test_prediction = bst_model_train_full.predict(data = full_testing_data)

    #for each cell, go through vec of probs and take index of max prob (if greater than ...): that's assignment

    
    test_predlabels = np.zeros((test_prediction.shape[0]))
    for i in range(test_prediction.shape[0]):
        if np.max(test_prediction[i, :]) > 1.1*(1/numbertrainclasses):
            test_predlabels[i] = np.argmax(test_prediction[i,:])
        
        #"unassigned" is a label one larger than all b/c python begins indexing at 0
        else:
            test_predlabels[i] = numbertrainclasses
        
    test_labels = np.zeros(len(test_anndata.obs[test_obs_id].values))
    for i,l in enumerate(test_anndata.obs[test_obs_id].values):
        test_labels[i] = test_dict[l]

    #actual labels of testing set, the labels that test set mapped to 
    return test_labels, test_predlabels, test_prediction

def calculateNCE(labels_true,labels_pred):
    X = labels_true
    Y = labels_pred
    contTable = confusion_matrix(X,Y)[0:len(np.unique(X)), 0:len(np.unique(Y))]
    a = np.sum(contTable, axis = 1)
    b = np.sum(contTable, axis = 0)
    N = np.sum(contTable)
    pij = contTable/N
    pi = a/N
    pj = b/N
    Hyx = np.zeros(contTable.shape)
    for i in range(contTable.shape[0]):
        for j in range(contTable.shape[1]):
          if pij[i,j] == 0:
            Hyx[i,j] = 0
          else:
            Hyx[i,j] = pij[i,j]*np.log10(pij[i,j]/pi[i])
    CE = -np.sum(Hyx)
    Hyi = np.zeros(contTable.shape[1])
    for j in range(contTable.shape[1]):
      if pj[j] == 0:
       Hyi[j] = 0
      else:
        Hyi[j] = pj[j]*np.log10(pj[j])
    Hy = -np.sum(Hyi)
    NCE = CE/Hy
    return NCE

def train_validate(adata, adata_cell,preproc=False):
    if (preproc):
        adata_cell.X = adata_cell.raw.X
        sc.pp.highly_variable_genes(adata_cell, min_mean=0.0125, max_mean=3, min_disp=0.5) #HVGs

    common_hvgs = list(set(adata[:,adata.var.highly_variable].var_names).intersection(set(adata_cell.var_names)))
    adata_m_dict = make_dict(adata, obs_id='leiden')
    adata_cell_dict = make_dict(adata_cell, obs_id='Subclass')
    print(len(common_hvgs), 'Shared HVGs')

    valid_truelabel_adata_cell, valid_predlabels_adata_cell, model_atlas_adata_cell = trainclassifier(train_anndata=adata_cell, 
                                                                                                 common_top_genes=common_hvgs, 
                                                                                                 obs_id='Subclass', 
                                                                                                 train_dict=adata_cell_dict, 
                                                                                                 eta=0.2,
                                                                                                 max_cells_per_ident=1000, 
                                                                                          train_frac=0.7, 
                                                                                                 min_cells_per_ident=100)

    plot_validation_plots(valid_truelabel_adata_cell, valid_predlabels_adata_cell, train_dict=adata_cell_dict)
    
    return model_atlas_adata_cell

def pairwise_map(adata_t0, adata_t1, test_lab, train_lab, t0_dict, t1_dict, recomp_HVGs, min_cells,
                x_lab, y_lab, union_hvgs):
    adata_t0 = adata_t0.copy() 
    adata_t1 = adata_t1.copy() 
    adata_t0.X = adata_t0.raw.X
    adata_t1.X = adata_t1.raw.X
    
    if (recomp_HVGs):
        sc.pp.highly_variable_genes(adata_t0, min_mean=0.0125, max_mean=3, min_disp=0.5) #HVGs
        sc.pp.highly_variable_genes(adata_t1, min_mean=0.0125, max_mean=3, min_disp=0.5) #HVGs
        t0_hvgs = list(adata_t0[:, adata_t0.var.highly_variable].var_names)
        t1_hvgs = list(adata_t1[:, adata_t1.var.highly_variable].var_names)
        
        if (union_hvgs):
            t0_t1hvgs_ = list(set(t0_hvgs).union(t1_hvgs))
            data_inter = list(set(adata_t1.var_names).intersection(set(adata_t0.var_names)))
            t0_t1hvgs = list(set(data_inter).intersection(set(t0_t1hvgs_)))

        else:
            t0_t1hvgs = list(set(t0_hvgs).intersection(t1_hvgs))

    
    else:
        t0_hvgs = list(adata_t0[:, adata_t0.var.highly_variable].var_names)
        t1_hvgs = list(adata_t1[:, adata_t1.var.highly_variable].var_names)
        t0_t1hvgs = list(set(t0_hvgs).intersection(t1_hvgs))

    print(len(t0_t1hvgs), 'shared HVGs')
    
    validation_label_train_70t0vst1, valid_predlabels_train_70t0vst1, model_t1,model_f1 = trainclassifier(train_anndata=adata_t1, 
                                                                                                 common_top_genes=t0_t1hvgs, 
                                                                                                 obs_id=train_lab, 
                                                                                                 train_dict=t1_dict, 
                                                                                                 eta=0.2,
                                                                                                 max_cells_per_ident=1000, 
                                                                                          train_frac=0.7, 
                                                                                                min_cells_per_ident=min_cells)

    plot_validation_plots(validation_label_train_70t0vst1, valid_predlabels_train_70t0vst1, train_dict=t1_dict)

    test_labelst0vst1, test_predlabelst0vst1, test_prediction_t0vst1 = predict(train_anndata=adata_t1, 
                                                                     common_top_genes=model_t1.feature_names, 
                                                                     bst_model_train_full=model_t1, 
                                                                     test_anndata=adata_t0,
                                                                     train_obs_id=train_lab, 
                                                                     test_dict=t0_dict, 
                                                                     test_obs_id=test_lab)

    mappingconfmatt0vst1, mappingxtickst0vst1, mappingplott0vst1 = plot_mapping(test_labels=test_labelst0vst1, 
                 test_predlabels=test_predlabelst0vst1, 
                 test_dict=t0_dict, 
                 train_dict=t1_dict,
                     xaxislabel=x_lab, yaxislabel=y_lab,)
    
    test_items = test_labelst0vst1, test_predlabelst0vst1, test_prediction_t0vst1
    mapping_items = mappingconfmatt0vst1, mappingxtickst0vst1, mappingplott0vst1
    
    return test_items, mapping_items

def make_colors(PX_dict):
    PX_colors = []
    for i in PX_dict:
        for j in colors_subclass:
            if (j in i):
                PX_colors.append(colors_subclass[j])
    return PX_colors

def rem_fem(adata):
    for j in list(adata.obs.Type_nn_dists.values.categories):
        if ('Fem' in j):
            adata = adata[adata.obs['Type_nn_dists']!=j,:]
            adata = adata[adata.obs['Type_leiden']!=j,:]
    return adata

def list2dict(list_in):

    adata_dict = {}
    for num,i in enumerate(list_in):
        adata_dict[i] = num
        
    #adata_dict['Unassigned'] = num + 1
    
    return adata_dict