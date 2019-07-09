# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, ShuffleSplit
from sklearn.feature_selection import RFECV
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import collections
from sklearn.metrics.pairwise import pairwise_distances

# Simple correlations of features
def cross_correlations(df1,df2,subject_ID_col):
    """ Computes correlation between features produced by two pipelines 
        Note: avg cross-correlations using .corr() diangonals because corrwith() doesn't seem to always work... 
    """
    n_roi = len(df1.columns)-1 #cols=subject_ID + roi_cols

    col_rename_dict = {} #Only need to rename on df
    
    for col in df1.columns:
        col_rename_dict[col] = str(col) + '_df2'

    df2 = df2.rename(columns=col_rename_dict)
    df2 = df2.rename(columns={'{}_df2'.format(subject_ID_col):subject_ID_col})

    concat_df = df1.merge(df2, on=subject_ID_col)
    #print('Shape of concatinated dataframe of two pipelines {}'.format(concat_df.shape))
    corr_mat = concat_df.corr()
    #print('Shape of corr mat {}'.format(corr_mat.shape))
    xcorr_dig = corr_mat.values[:n_roi,n_roi:2*n_roi].diagonal()
    xcorr_df = pd.DataFrame(columns=['ROI','correlation'])
    xcorr_df['ROI'] = corr_mat.columns[:n_roi] 
    xcorr_df['correlation'] = xcorr_dig

    return xcorr_df

# Significance of correlation between ROIs (within a given pipeline)
def calculate_pvalues(df):
    """ computes p values of correlation between ROIs in a dataframe format (sub x roi) 
    """
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

# cluster memberships
def get_cluster_membership(_df,g,n_clusters):
    member_df = _df[['SubjID']].copy()
    Z = g.dendrogram_row.linkage
    T = sch.fcluster(Z, n_clusters, 'maxclust')
    member_count = collections.Counter(T)
    pipe = _df['pipeline'].values[0]
    print('Pipeline {}, cluster sizes {}'.format(pipe, member_count))
    member_df['membership_{}'.format(pipe)] = T
    return member_df

def generate_pairwise_membership(df,m_col):
    membership = df[m_col].values
    # Just want to know if two subjects are in the same cluster (hamming will compute the inverse)
    pairwise_membership = 1-pairwise_distances(np.array(membership).reshape(-1,1),metric='hamming')
    df_membership = pd.DataFrame(columns=df['SubjID'], index=df['SubjID'],data=pairwise_membership)
    return df_membership

    
# ML model perfs
def computePipelineMLModels(df,roi_cols,covar_continuous_cols,covar_cat_cols,outcome_col,model_type,ml_model,rank_features,n_splits=10,n_repeats=10,n_jobs=1):
    """ Compares performance of different pipeline outputs for a given ML Model
        Calls getMLModelPerf to get individual model performances
    """
    pipelines = df['pipeline'].unique()
    print('Running ML classifer on {} pipelines'.format(len(pipelines)))
    scores_concat_df = pd.DataFrame()
    feature_rank_concat_df = pd.DataFrame()
    perf_pval_dict = {}
    for pipe in pipelines:
        ml_df = df[df['pipeline']==pipe]
        print('Pipeline {}'.format(pipe))
        scores_df, null_df, pvalue, feature_rank_df = getMLModelPerf(ml_df,roi_cols,covar_continuous_cols,covar_cat_cols,outcome_col,model_type,ml_model,rank_features,n_splits,n_repeats,n_jobs)    
        scores_df['pipeline'] = np.tile(pipe,len(scores_df))
        null_df['pipeline'] = np.tile('null',len(null_df))
        scores_concat_df = scores_concat_df.append(scores_df).append(null_df)
        perf_pval_dict[pipe] = pvalue
        feature_rank_df['pipeline'] = np.tile(pipe,len(feature_rank_df))
        feature_rank_concat_df = feature_rank_concat_df.append(feature_rank_df)
    return scores_concat_df, perf_pval_dict, feature_rank_concat_df

def getMLModelPerf(ml_df,roi_cols,covar_continuous_cols,covar_cat_cols,outcome_col,model_type,ml_model,rank_features,n_splits=10,n_repeats=10,n_jobs=1):
    """ Takes a model (classification or regression) instance and computes cross val scores.
        Uses repeated stratified KFold for classification and ShuffeSplit for regression.
    """     
    X = ml_df[roi_cols].values
    X_col_names = roi_cols.copy()
    # Check input var types and create dummy vars if needed
    if len(covar_continuous_cols) > 0:
        X_continuous_covar = ml_df[covar_continuous_cols].values
        print('Using {} continuous covar'.format(len(covar_continuous_cols)))
        X = np.hstack((X, X_continuous_covar))
        X_col_names += list(covar_continuous_cols)
    if len(covar_cat_cols) > 0:
        X_cat_covar_df = pd.get_dummies(ml_df[covar_cat_cols])
        X_cat_covar = X_cat_covar_df.values
        print('Using {} categorical cols as {} cat covar (dummies)'.format(covar_cat_cols,X_cat_covar.shape[1]))
        X = np.hstack((X, X_cat_covar))
        X_col_names += list(X_cat_covar_df.columns)

    print('n of input columns: {}'.format(len(X_col_names)))
    if model_type.lower() == 'classification':
        y = pd.get_dummies(ml_df[outcome_col]).values[:,0]
        print('Data shapes X {}, y {} ({})'.format(X.shape, len(y), list(ml_df[outcome_col].value_counts())))  
        perf_metric = 'roc_auc'
        cv = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
    elif model_type.lower() == 'regression':
        y = ml_df[outcome_col].values
        print('Data shapes X {}, y {} ({:3.2f}m, {:3.2f}sd)'.format(X.shape, len(y), np.mean(y),np.std(y)))   
        perf_metric = 'neg_mean_squared_error'
        cv = ShuffleSplit(n_splits=n_splits*n_repeats, random_state=0)
    else:
        print('unknown model type {} (needs to be classification or regression)'.format(model_type))

    print('Using {} model with perf metric {}'.format(model_type, perf_metric))
    perf = cross_val_score(ml_model, X, y, scoring=perf_metric,cv=cv, n_jobs=n_jobs)
    scores_df = pd.DataFrame(columns=[perf_metric])
    scores_df[perf_metric] = perf
    print(' Perf mean:{:4.3f}, sd:{:4.3f}'.format(np.mean(perf),np.std(perf)))

    # Null model 
    null_cv = ShuffleSplit(n_splits=n_repeats, random_state=0) #10x10xn_permutations are too many. 
    _, permutation_scores, pvalue = permutation_test_score(ml_model, X, y, scoring=perf_metric, cv=null_cv, n_permutations=10, n_jobs=n_jobs)
    null_df = pd.DataFrame()
    null_df[perf_metric] = permutation_scores

    # Feature ranks based on RFECV
    feature_ranks_df = pd.DataFrame()
    if rank_features:
        feature_ranks, feature_grid_scores = get_feature_importance(ml_model, X, y, perf_metric, cv=cv, n_jobs=n_jobs)
        feature_ranks_df['predictor'] = X_col_names
        feature_ranks_df['rank'] = feature_ranks
        feature_ranks_df['grid_scores'] = feature_grid_scores

    return scores_df, null_df, pvalue, feature_ranks_df

def get_feature_importance(model, X, y, perf_metric, n_jobs, step=1, cv=5):
    selector = RFECV(model, step=1, scoring=perf_metric, cv=cv, n_jobs=n_jobs)
    selector = selector.fit(X, y)
    feature_ranks = selector.ranking_
    feature_grid_scores = selector.grid_scores_
    return feature_ranks, feature_grid_scores

# Stat model perfs

def getCorrectedPValues(pval_raw,alpha=0.05,method='fdr_i'):
    rej, pval_corr = smm.multipletests(pval_raw, alpha=alpha, method=method)[:2]
    return pval_corr

def computePipelineStatsModels(df,roi_cols,covar_cols,outcome_col,signific_col,stat_model):
    """ Compares performance of different pipeline outputs for a given ML Model
        Calls getStatModelPerf to get individual model performances
    """
    pipelines = df['pipeline'].unique()
    print('Running {} mass-univariate {} statsmodels on {} pipelines'.format(len(roi_cols), stat_model, len(pipelines)))
    
    # index results on ROI names
    scores_concat_df = pd.DataFrame()
    for pipe in pipelines:
        sm_df = df[df['pipeline']==pipe]
        print('Pipeline {}'.format(pipe))
        scores_df = getStatModelPerf(sm_df,roi_cols,covar_cols,outcome_col,signific_col,stat_model)
        scores_df['pipeline'] = np.tile(pipe,len(scores_df))
        scores_concat_df = scores_concat_df.append(scores_df)
        print('Top 10 significant regions:\n {}'.format(scores_df.sort_values(by=['p_val']).head(10)))

    return scores_concat_df

def getStatModelPerf(sm_df,roi_cols,covar_cols,outcome_col,signific_col,stat_model):
    """ Creates either a OLS or Logit instance and computes t_vals, p_vals, model fit etc. 
        Does not support other models at the moment since you cannot pass an instance 
        of a stats_models without providing X and Y. 
    """
    
    outcome_col_val = outcome_col[0]
    outcome_col_type = outcome_col[1]
    model_name_check = False
    if (stat_model.lower() == 'logit') and (outcome_col_type=='cat'):
        model_name_check = True
    elif (stat_model.lower() == 'ols') and (outcome_col_type=='continuous'):
        model_name_check = True
    else:
        print('Outcome var type {} and stats model {} mismatch'.format(outcome_col_type,stat_model))
        model_name_check = False

    if model_name_check:
        scores_df = pd.DataFrame(columns= ['roi','coef','t_val','p_val','p_val_corr'])
        coef_list = []
        t_val_list = []
        p_val_list = []
      
        signific_col_val = signific_col[0]
        signific_col_type = signific_col[1]

        covar_val_list = []
        covar_string = ''
        if len(covar_cols) > 0:
            for covar in covar_cols:
                covar_val = covar[0]
                covar_type = covar[1]
                covar_val_list.append(covar_val)
                if covar_type == 'cat':
                    covar_string = covar_string + ' + C({})'.format(covar_val)
                elif covar_type == 'continuous':
                    covar_string = covar_string + ' + {}'.format(covar_val)
                else:
                    print('unknown covar type {}'.format(covar))

        for roi in roi_cols:
            if outcome_col_val != 'roi': #using roi as predictor (independent var)
                signific_col_val = roi
                input_cols = [outcome_col_val, roi] + covar_val_list
                X = sm_df[input_cols]
                formula_string = '{} ~ {}{}'.format(outcome_col_val,roi,covar_string)
                stat_summary_name = signific_col_val
            else: # using roi as outcome (dependent var)
                input_cols = [roi, signific_col_val] + covar_val_list
                X = sm_df[input_cols]
                if signific_col_type == 'continuous':
                    formula_string = '{} ~ {}{}'.format(roi,signific_col_val,covar_string)
                    stat_summary_name = signific_col_val
                elif signific_col_type == 'cat':
                    formula_string = '{} ~ C({}){}'.format(roi,signific_col_val,covar_string)
                    stat_summary_name = 'C({})[T.1]'.format(signific_col_val)
                else:
                    print('unknown signficance independent variable type: {}'.format(signific_col))
           
            if stat_model.lower() == 'logit':
                model = smf.logit(formula=formula_string,data=X)

            elif stat_model.lower() == 'ols':
                model = smf.ols(formula=formula_string,data=X)

            else:
                print('Unknown stats model {}'.format(stat_model))

            # Can't do fit_regularized for mass-univariate approach. 
            results = model.fit(disp=0) #default newton fails for smaller N (even smaller site N)
            
            coef = results.params[stat_summary_name] # just for ROI
            t_val = results.tvalues[stat_summary_name] # just for ROI
            p_val = results.pvalues[stat_summary_name] # just for ROI
            coef_list.append(coef)
            t_val_list.append(t_val)
            p_val_list.append(p_val)

        #FDR Correction
        p_val_corr_list = getCorrectedPValues(p_val_list)
        print('Example statsmodel run:\n {}'.format(formula_string))

        scores_df['roi'] = roi_cols
        scores_df['coef'] = coef_list
        scores_df['t_val'] = t_val_list
        scores_df['p_val'] = p_val_list
        scores_df['p_val_corr'] = p_val_corr_list

    return scores_df

def aggregate_perf(df,measure,thresh=0.05):
    """ Aggregates performance from different pipeline variations (tools, atlases)
        Currently aggregates using simple ranking 
    """
    df_agg = pd.DataFrame(columns=['roi','rank'])
    df['significance'] = df[measure] < thresh
    roi_list = df['roi'].unique()
    rank_list = []
    for roi in roi_list:
        rank_list.append(np.sum(df[df['roi']==roi]['significance'].values))
    df_agg['roi'] = roi_list
    df_agg['rank'] = rank_list
    
    return df_agg
