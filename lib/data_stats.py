# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import statsmodels.api as sm

def cross_correlations(df1,df2,subject_ID_col):
    """ Computes correlation between features produced by two pipelines 
        Note: avg cross-correlations using .corr() diangonals because corrwith() doesn't seem to always work... 
    """
    n_roi = len(df1.columns)-1 #cols=subject_ID + roi_cols

    col_rename_dict_1 = {}
    col_rename_dict_2 = {}
    
    for col in df1.columns:
        col_rename_dict_1[col] = str(col) + '_df1' 
        col_rename_dict_2[col] = str(col) + '_df2'

    df1 = df1.rename(columns=col_rename_dict_1)
    df2 = df2.rename(columns=col_rename_dict_2)
    df1 = df1.rename(columns={'{}_df1'.format(subject_ID_col):subject_ID_col})
    df2 = df2.rename(columns={'{}_df2'.format(subject_ID_col):subject_ID_col})

    concat_df = df1.merge(df2, on=subject_ID_col)
    #print('Shape of concatinated dataframe of two pipelines {}'.format(concat_df.shape))
    corr_mat = concat_df.corr()
    #print('Shape of corr mat {}'.format(corr_mat.shape))
    xcorr = corr_mat.values[:n_roi,n_roi:2*n_roi].diagonal()

    return xcorr


def getClassiferPerf(df,input_cols,outcome_col,clf,n_splits=10,n_repeats=10):
    """ Takes a classifier instance and computes cross val scores on repeated stratified KFold
        on all pipelines listed in the df
    """
    pipelines = df['pipeline'].unique()
    print('Running ML classifer on {} pipelines'.format(len(pipelines)))
    scores_concat_df = pd.DataFrame(columns=['pipeline','Acc'])
    for pipe in pipelines:
        ml_df = df[df['pipeline']==pipe]
        X = ml_df[input_cols].values
        y = pd.get_dummies(ml_df[outcome_col]).values[:,0]
        acc = cross_val_score(clf, X, y, cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats))
        scores_df = pd.DataFrame(columns=['pipeline','Acc'])
        scores_df['pipeline'] = np.tile(pipe,len(acc))
        scores_df['Acc'] = acc
        scores_concat_df = scores_concat_df.append(scores_df)
        print('Pipeline {},  Accuracy mean:{:4.3f}, sd:{:4.3f}'.format(pipe,np.mean(acc),np.std(acc)))
    return scores_concat_df    

def getStatModelPerf(df,roi_cols,covar_cols,outcome_col,stat_model):
    """ Creates either a OLS or Logit instance and computes t_vals, p_vals, model fit etc. 
        Does not support other models at the moment since you cannot pass an instance 
        of a stats_models without providing X and Y. 
    """
    
    model_name_check = True
    if not stat_model.lower() in ['logit','ols']:
        print('Unknown stats model')
        model_name_check = False

    if model_name_check:
        pipelines = df['pipeline'].unique()
        print('Running {} mass-univariate {} statsmodels on {} pipelines'.format(len(roi_cols), stat_model, len(pipelines)))
        
        # index results on ROI names
        scores_concat_df = pd.DataFrame(columns= ['roi','pipeline','t_val','p_val'])
        for pipe in pipelines:
            sm_df = df[df['pipeline']==pipe]
            scores_df = pd.DataFrame(columns= ['roi','pipeline','t_val','p_val'])
            t_val_list = []
            p_val_list = []
            for roi in roi_cols:
                input_cols = [roi] + covar_cols
                X = sm_df[input_cols]
                if stat_model.lower() == 'logit':
                    y = pd.get_dummies(sm_df[outcome_col]).values[:,0]
                    model = sm.Logit(y,X)
                elif stat_model.lower() == 'ols':
                    y = sm_df[outcome_col].values
                    model = sm.OLS(y,X)

                results = model.fit(disp=0)
                t_val = results.tvalues[0] # just for ROI
                p_val = results.pvalues[0] # just for ROI
                t_val_list.append(t_val)
                p_val_list.append(p_val)

            scores_df['roi'] = roi_cols
            scores_df['pipeline'] = np.tile(pipe, len(roi_cols))
            scores_df['t_val'] = t_val_list
            scores_df['p_val'] = p_val_list
            scores_concat_df = scores_concat_df.append(scores_df)

    return scores_concat_df

            

