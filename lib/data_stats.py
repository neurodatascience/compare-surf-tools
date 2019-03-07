# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 1 Feb 2019

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


def getClassiferPerf(df,roi_cols,covar_continuous_cols,covar_cat_cols,outcome_col,clf,n_splits=10,n_repeats=10):
    """ Takes a classifier instance and computes cross val scores on repeated stratified KFold
        on all pipelines listed in the df
    """
    pipelines = df['pipeline'].unique()
    print('Running ML classifer on {} pipelines'.format(len(pipelines)))
    scores_concat_df = pd.DataFrame(columns=['pipeline','Acc'])
    for pipe in pipelines:
        ml_df = df[df['pipeline']==pipe]
        X = ml_df[roi_cols].values

        #TODO handle covariates 
        if len(covar_continuous_cols) > 0:
            X_continuous_covar = ml_df[covar_continuous_cols].values
            print('Using {} continuous covar'.format(len(covar_continuous_cols)))
            X = np.hstack((X, X_continuous_covar))
        if len(covar_cat_cols) > 0:
            X_cat_covar = pd.get_dummies(ml_df[covar_cat_cols]).values
            print('Using {} col for {} cat covar'.format(len(covar_cat_cols),X_cat_covar.shape[1]))
            X = np.hstack((X, X_cat_covar))

        y = pd.get_dummies(ml_df[outcome_col]).values[:,0]

        print('Data shapes X {}, y {} ({})'.format(X.shape, len(y), list(ml_df[outcome_col].value_counts())))
        acc = cross_val_score(clf, X, y, cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats))
        scores_df = pd.DataFrame(columns=['pipeline','Acc'])
        scores_df['pipeline'] = np.tile(pipe,len(acc))
        scores_df['Acc'] = acc
        scores_concat_df = scores_concat_df.append(scores_df)
        print('Pipeline {},  Accuracy mean:{:4.3f}, sd:{:4.3f}'.format(pipe,np.mean(acc),np.std(acc)))
    return scores_concat_df    




def getStatModelPerf(df,roi_cols,covar_continuous_cols,covar_cat_cols,outcome_col,stat_model):
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
            covar_string = ''
            if len(covar_continuous_cols) > 0:
                for covar in covar_continuous_cols:
                    covar_string = covar_string + ' + {}'.format(covar)

            if len(covar_cat_cols) > 0:
                for covar in covar_cat_cols:
                    covar_string = covar_string + ' + C({})'.format(covar)

            for roi in roi_cols:
                input_cols = [outcome_col, roi] + covar_continuous_cols + covar_cat_cols
                X = sm_df[input_cols]
                formula_string = '{} ~ {}{}'.format(outcome_col,roi,covar_string)

                if stat_model.lower() == 'logit':
                    model = smf.logit(formula=formula_string,data=X)

                elif stat_model.lower() == 'ols':
                    model = smf.ols(formula=formula_string,data=X)

                results = model.fit(disp=0)
                t_val = results.tvalues[roi] # just for ROI
                p_val = results.pvalues[roi] # just for ROI
                t_val_list.append(t_val)
                p_val_list.append(p_val)

            print('Example statsmodel run:\n {}'.format(formula_string))

            scores_df['roi'] = roi_cols
            scores_df['pipeline'] = np.tile(pipe, len(roi_cols))
            scores_df['t_val'] = t_val_list
            scores_df['p_val'] = p_val_list
            scores_concat_df = scores_concat_df.append(scores_df)

    return scores_concat_df

            

