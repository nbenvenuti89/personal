# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:06:53 2020

@author: niccolo
"""

#from keras import Sequential
#from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
#import random
#from keras.wrappers.scikit_learn import KerasClassifier
import sklearn
import warnings; warnings.simplefilter('ignore')
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import fnmatch
import xgboost as xgb

os.chdir('C://Users/niccolo/Desktop/LTV/')



#----------------------------------#
# Permutation Features Importance
#----------------------------------#
def PermutationFeaturesImportance(df, X, predictions, nn_model, cut, kpi='F1Score', max_num_features=20, ax=None):
    
    X_test_pd = pd.DataFrame(data = df, columns = [x for x in X])
    benchmark = pd.DataFrame(columns = ['Feature','Accuracy','Precision','Recall','F1Score'])
    for i in X_test_pd: 
        #print(i)
        df = X_test_pd.copy()
        df[i] = 0
        prediction_temp = nn_model.predict(df)
        preds = prediction_temp/np.max(prediction_temp)
        preds_cut = 1* (preds>cut)
        conf_mat = sklearn.metrics.confusion_matrix(predictions,preds_cut)
        tn, fp, fn, tp = conf_mat.ravel()
        Accuracy, Precision, Recall= (tn+tp)/(tn+tp+fn+tp), tp/(tp+fp), tp/(fn+tp)
        F1Score = 2*((Precision*Recall)/(Precision+Recall))
        benchmark = benchmark.append(pd.DataFrame({'Feature': [i], 'Accuracy': [Accuracy], 'Precision': [Precision], 'Recall':[Recall], 'F1Score':[F1Score]}))
    # Plot
    benchmark = benchmark.sort_values(kpi).reset_index(drop=True)
    benchmark = benchmark.head(max_num_features)
    if ax is None:
        ax = plt.gca()    
    ax.barh(np.arange(len(benchmark.Feature)), (1-benchmark['F1Score']), tick_label=1)
    ax.set_yticks(np.arange(len(benchmark.Feature)))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_yticklabels(benchmark.Feature)
    ax.set_xlim(0.2,0.5)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_title('PermutationFeaturesImportance - NN', size=15)
    
    return ax, benchmark
    
    


def AppendingSources(rep, payers=True):
    tables = []
    string = {True: 'LTV_payers_data', False: 'LTV_non_payers_data'}
    for file in os.listdir(rep):
        if (fnmatch.fnmatch(file, '*'+str(string[payers])+'*')):
            tables.append(file)
    print(tables)
    for i in range(len(tables)):
        print(str(i+1)+'\n'+str(tables[i]))
        if i == 0:
            payers_data = pd.read_pickle(str(rep)+str(tables[i]))
        else:
            payers_data = payers_data.append(pd.read_pickle(str(rep)+str(tables[i])))
    print(payers_data.shape)
    payers_data.to_pickle(str(rep)+str(string[payers])+'.pkl')



#----------------------------------#
# HyperParameter Optimizer
#----------------------------------#
def HypParams_Optimizer(X_train, Target):

    from keras.wrappers.scikit_learn import KerasClassifier
    from keras import Sequential
    from keras.layers import Dense
    
    def build_classifier(optimizer):
        classifier = Sequential()
        classifier.add(Dense(units=35, kernel_initializer='uniform', activation='relu', input_dim = 69))
        classifier.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=8,  kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier    
    classifier = KerasClassifier(build_fn = build_classifier) # batch_size and epochs will be in the grid search
    parameters = {'batch_size': [1024, 4096],
                  'epochs': 100,
                  'optimizer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'f1', # f1 for F1 score, accuracy
                               cv = 5)
    grid_search = grid_search.fit(X_train, Target)
    return grid_search.best_params_, grid_search.best_score_


#----------------------------------#
# Plot ROC Curve
#----------------------------------#
def ROC_Curve(nn_model, X_test, target, ax=None):
    y_pred_keras = nn_model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(target, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    if ax is None:
        ax = plt.gca()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    ax.set(xlabel='False positive rate', ylabel='True positive rate')
    ax.set_title('ROC curve')
    ax.legend(loc='best')
    return(ax)
    
    
#----------------------------------#
# Best F1-Score Prediction CutOff
#----------------------------------#    
def BestPredCut(df,Y_Predicted, metric='F1_Score'):
    Best_Score=0
    for i in np.arange(0.10, 0.9, .01):
        df['preds_cut'] = 1* (df['preds']>i)
        conf_mat = sklearn.metrics.confusion_matrix(df[Y_Predicted],df[['preds_cut']])
        tn, fp, fn, tp = conf_mat.ravel()
        Accuracy, Precision, Recall= (tn+tp)/(tn+tp+fn+fp), tp/(tp+fp), tp/(fn+tp)
        F1_Score = 2*((Precision*Recall)/(Precision+Recall))
        AUC = roc_auc_score(df[Y_Predicted],df[['preds']])
        score = eval(metric)
        if score>Best_Score:
            Best_Score = score
            cut = i
            next
        else:
            next
    return AUC, cut

   
    
#----------------------------------#
# Cross Entropy
#----------------------------------#    
def CrossEntropy(x):
    cross_ent = - (x['STAYED_ACTIVE'] * np.log(x['preds']) + (1-x['STAYED_ACTIVE']) * np.log(1-x['preds']))
    return np.mean(cross_ent)

    
#----------------------------------#
# Classification Metrics by Group
#----------------------------------#    
def ClassGroup(x, col):
    conf_mat = sklearn.metrics.confusion_matrix(x[['STAYED_ACTIVE']],x[['preds_cut']])
    tn, fp, fn, tp = conf_mat.ravel()
    exp_rev = np.sum(x['expected_rev_cut']) / float(np.sum(x[col]))        
    return [(tn+tp)/(tn+tp+fn+fp), tp/(tp+fp), tp/(fn+tp), 2*(((tp/(tp+fp))*(tp/(fn+tp)))/((tp/(tp+fp))+(tp/(fn+tp)))), exp_rev, len(x)]

    
#----------------------------------#
# Regression Metrics by Group
#----------------------------------#  
def RegGroup(x, col):
    y_pred_rev = np.sum(x['Prediction'])
    y_test_rev = np.sum(x[col])
    std = np.std(x['Prediction'] - x[col])
    count = len(x)
    return y_test_rev, y_pred_rev, std, count
    
    
    
#----------------------------------#
# Moving Average
#----------------------------------#    
def MovingAverage(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    
#----------------------------------#
# Natural Log
#----------------------------------#    
def Logn(x):
    logn = np.log(np.where(x==0, 0.1, x))
    return logn

    
#----------------------------------#
# Square Root
#----------------------------------#    
def SQRT(x):
    sqrt = np.sqrt(x)
    return sqrt
    
    
#----------------------------------#
# Features Scaling
#----------------------------------#    
def FeaturesScaling(Scaling, X):
    if Scaling == 'Norm':
        scaler = MinMaxScaler() 
        scaler.fit(X)
        #print(scaler.data_max_)
        #print(scaler.data_min_)
        Scaled_X = scaler.transform(X) 
    elif Scaling == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X)
        #print(scaler.mean_)
        #print(scaler.var_)
        Scaled_X = scaler.transform(X) 
    return Scaled_X, scaler
    
    
#----------------------------------#
# Extract Active Predictive Players
#----------------------------------# 
def ExtractPayers(model, df, matrix, cut):
    df['Pred_Prob']= model.predict(matrix)
    df['Pred_Class'] = 1* (df['Pred_Prob']>=cut)
    df = df[df.Pred_Class==1]
    return df

    
#----------------------------------#
# Target Analysis
#----------------------------------# 
def TargetAnalysis(x):
    targetmean = np.mean(x['STAYED_ACTIVE'])
    count = len(x)
    return targetmean, count



#----------------------------------#
# Data Cleaner 
# Drop Features and get Target
#----------------------------------# 
def DataCleaner(df, MonthPrediction, Segment, Non_Balance, Target):
    Features2Exclude = []
    FeaturesPredict = {3: {'Revenue' :'TOTAL_REVENUE_NEXT3MONTHS', 'Purchases':'TOTAL_PURCHASES_NEXT3MONTHS', 'SkipMonths': []}, 
                       6: {'Revenue' :'TOTAL_REVENUE_NEXT6MONTHS', 'Purchases':'TOTAL_PURCHASES_NEXT6MONTHS', 'SkipMonths': []},
                       9: {'Revenue' :'TOTAL_REVENUE_NEXT9MONTHS', 'Purchases':'TOTAL_PURCHASES_NEXT9MONTHS', 'SkipMonths': ['October19','November19', 'December19']}, 
                       12:{'Revenue' :'TOTAL_REVENUE_NEXT12MONTHS','Purchases':'TOTAL_PURCHASES_NEXT12MONTHS','SkipMonths': ['July19','August19','September19','October19','November19','December19']},
                       14:{'Revenue' :'TOTAL_REVENUE_NEXT14MONTHS','Purchases':'TOTAL_PURCHASES_NEXT14MONTHS','SkipMonths': ['May19','June19','July19','August19','September19','October19','November19','December19']},
                       16:{'Revenue' :'TOTAL_REVENUE_NEXT16MONTHS','Purchases':'TOTAL_PURCHASES_NEXT16MONTHS','SkipMonths': ['March19','April19','May19','June19','July19','August19','September19','October19','November19','December19']},
                       18:{'Revenue' :'TOTAL_REVENUE_NEXT18MONTHS','Purchases':'TOTAL_PURCHASES_NEXT18MONTHS','SkipMonths': ['January19','February19','March19','April19','May19','June19','July19','August19','September19','October19','November19','December19']},
                       21:{'Revenue' :'TOTAL_REVENUE_NEXT21MONTHS','Purchases':'TOTAL_PURCHASES_NEXT21MONTHS','SkipMonths': ['October18','November18', 'December18','January19','February19','March19','April19','May19','June19','July19','August19','September19','October19','November19','December19']},
                       24:{'Revenue' :'TOTAL_REVENUE_NEXT24MONTHS','Purchases':'TOTAL_PURCHASES_NEXT24MONTHS','SkipMonths': ['July18','August18','September18','October18','November18', 'December18','January19','February19','March19','April19','May19','June19','July19','August19','September19','October19','November19','December19']}}
    for key, value in FeaturesPredict.items():
        Features2Exclude.append(value['Revenue'])
        Features2Exclude.append(value['Purchases'])
    
    if Segment == 'non_payers':
        listNoPayers = ['FTD_AGE', 'LAST_PURCHASE_LASTACTIVEDATE', 'SUM_REV_TODAY', 'SUM_TRX_TODAY', 'MAX_REV_TODAY', 'MAX_TRX_TODAY','AVGPURCHASE', 'MEDIANPURCHASE', 'DAYSMONETIZER',
                        'NUM_PURCHASES_LAST3_DAYS','NUM_PURCHASES_LAST7_DAYS','NUM_PURCHASES_LAST14_DAYS','NUM_PURCHASES_LAST_MONTH','NUM_PURCHASES_LAST3MONTHS',
                        'TOT_REVENUE_LAST3_DAYS','TOT_REVENUE_LAST7_DAYS','TOT_REVENUE_LAST14_DAYS','TOT_REVENUE_LAST_MONTH','TOT_REVENUE_LAST3MONTHS','LAST_PURCHASE_MAXDATEPURCHASE',
                        'NUM_PURCHASES_FIRST3_DAYS','NUM_PURCHASES_FIRST7_DAYS','NUM_PURCHASES_FIRST30_DAYS','TOT_REVENUE_FIRST3_DAYS','TOT_REVENUE_FIRST7_DAYS','TOT_REVENUE_FIRST30_DAYS']   
        for feat in listNoPayers:
            Features2Exclude.append(feat)
    list2exclude = ['PERIOD','PLAYERID','FIRST_INSTALL','LAST_INSTALL','DATE_MONTH','CHANNEL','CAMPAIGN_NAME','DATE_YEAR',
                    'CAMPAIGN_ID','ADGROUPNAME','CREATIVENAME','ADID', 'ADJID', 'FIRST_DID', 'LAST_DID', 'STAYED_ACTIVE',
                    'COUNTRY','PLATFORM','INSTALL_SOURCE','SOURCE_TYPE','SOURCE_VERTICAL','FACEBOOKID']
    for i in Features2Exclude: list2exclude.append(i)
    ExtraFeatures =  ['LP_BALANCE','LP_BALANCE_LAST_7_DAYS','LP_BALANCE_LAST_1_MONTH','LP_BALANCE_LAST_3_MONTH',
                      'CHIP_BALANCE','CHIP_BALANCE_LAST_7_DAYS','CHIP_BALANCE_LAST_1_MONTH','CHIP_BALANCE_LAST_3_MONTH',
                      #'SUM_SPINS','SUM_SPINS_LAST_7_DAYS','SUM_SPINS_LAST_1_MONTH','SUM_SPINS_LAST_3_MONTH',
                      'AVGPURCHASE']
    if Non_Balance:
        for j in ExtraFeatures: list2exclude.append(j)
    FeaturesPredict, SkipMonths = FeaturesPredict[MonthPrediction][Target], FeaturesPredict[MonthPrediction]['SkipMonths']
    
    return FeaturesPredict, list2exclude, SkipMonths



#----------------------------------#
# Create dummies to be replicated
#----------------------------------# 
def GetDummies(df, fixed):
    if fixed:
        df.COUNTRY = np.where((~df.COUNTRY.isin(['US','CA','GB']))|(df.COUNTRY.isna()), 'Others', df.COUNTRY)
        df['COUNTRY_CA'] = np.where(df.COUNTRY == 'CA', 1, 0)
        df['COUNTRY_US'] = np.where(df.COUNTRY == 'US', 1, 0)
        df['COUNTRY_GB'] = np.where(df.COUNTRY == 'GB', 1, 0)
        df['PLATFORM_IOS'] = np.where(df.PLATFORM == 'ios', 1, 0)
        df['PLATFORM_GPS'] = np.where(df.PLATFORM == 'android', 1, 0)
        df['PLATFORM_AMZ'] = np.where(df.PLATFORM == 'amazon', 1, 0)
        df['INSTALL_SOURCE_ORGANIC'] = np.where(df.INSTALL_SOURCE == 'ORGANIC', 1, 0)
        df['INSTALL_SOURCE_PAID'] = np.where(df.INSTALL_SOURCE == 'PAID', 1, 0)
        df['INSTALL_SOURCE_XPROMO'] = np.where(df.INSTALL_SOURCE == 'XPROMO', 1, 0)
        df['SOURCE_TYPE_NOT_PAID'] = np.where(df.SOURCE_TYPE == 'NOT_PAID', 1, 0)
        df['SOURCE_TYPE_INCENT'] = np.where(df.SOURCE_TYPE == 'INCENT', 1, 0)
        df['SOURCE_TYPE_NONINCENT'] = np.where(df.SOURCE_TYPE == 'NONINCENT', 1, 0)
        df['SOURCE_VERTICAL_NOT_PAID'] = np.where(df.SOURCE_VERTICAL == 'NOT_PAID', 1, 0)
        df['SOURCE_VERTICAL_ORGANIC'] = np.where(df.SOURCE_VERTICAL == 'ORGANIC', 1, 0)
        df['SOURCE_VERTICAL_NETWORK'] = np.where(df.SOURCE_VERTICAL == 'NETWORK', 1, 0)
        df['SOURCE_VERTICAL_SANS'] = np.where(df.SOURCE_VERTICAL == 'SANS', 1, 0)
        df['SOURCE_VERTICAL_PROGR'] = np.where(df.SOURCE_VERTICAL == 'PROGRAMMATIC', 1, 0)
        #df.drop(['COUNTRY','PLATFORM','INSTALL_SOURCE','SOURCE_TYPE','SOURCE_VERTICAL'],axis=1, inplace=True) 
    else:
        df = pd.concat([df, pd.get_dummies(df.COUNTRY, prefix='COUNTRY', drop_first=True)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.PLATFORM, prefix='PLATFORM', drop_first=True)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.INSTALL_SOURCE, prefix='INSTALL_SOURCE', drop_first=True)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.SOURCE_TYPE, prefix='SOURCE_TYPE', drop_first=True)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.SOURCE_VERTICAL, prefix='SOURCE_VERTICAL', drop_first=True)], axis=1)
        #df.drop(['COUNTRY','PLATFORM','INSTALL_SOURCE','SOURCE_TYPE','SOURCE_VERTICAL'],axis=1, inplace=True) 
    
    return df

    
#----------------------------------#
# Custom Eval for XGBoost
#----------------------------------# 
def custom_asymmetric_train(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    residual = (y_true - y_pred)
    grad = np.where(residual>0, -2*1.6*residual, -2*residual)
    hess = np.where(residual>0, 2*2.5, 2.0)
    return grad, hess

def custom_asymmetric_eval(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    residual = (y - predt)
    w_res = np.where(residual>0, 1.6*residual, residual)
    elements = np.power(w_res, 2)
    return 'eval_err',np.sqrt(float(np.sum(elements) / len(y)))
    
    
#----------------------------------#
# Assigns 1 to Outliers
#----------------------------------# 
def OutliersDetection(x, df1, FeaturesPredict, scale):
    df1['std'] = np.std(x[FeaturesPredict][x[FeaturesPredict]>0])
    df1['mean'] = np.mean(x[FeaturesPredict][x[FeaturesPredict]>0])
    df1['Outlier'] = np.where(df1[FeaturesPredict]>df1['mean']+scale*df1['std'], 1, 0)
    return df1
    
    
    
#----------------------------------#
# Principal Components Analysis
# for Install source information
#----------------------------------# 
def PCA_Attribution(X, n_components = 3, Save=True):
    import pickle
    from sklearn.decomposition import PCA
    PCA_matrix = X[['COUNTRY_CA','COUNTRY_US', 'COUNTRY_GB', 'PLATFORM_IOS', 'PLATFORM_GPS','PLATFORM_AMZ', 
                    'INSTALL_SOURCE_ORGANIC', 'INSTALL_SOURCE_PAID','INSTALL_SOURCE_XPROMO', 'SOURCE_TYPE_NOT_PAID', 
                    'SOURCE_TYPE_INCENT','SOURCE_TYPE_NONINCENT', 'SOURCE_VERTICAL_NOT_PAID','SOURCE_VERTICAL_ORGANIC', 
                    'SOURCE_VERTICAL_NETWORK','SOURCE_VERTICAL_SANS', 'SOURCE_VERTICAL_PROGR']]
    pca = PCA(n_components = n_components) # None for all
    X_PCA = pca.fit_transform(PCA_matrix)
    #print(pca.explained_variance_ratio_)
    #print(pca.explained_variance_)
    for i in range(n_components):
        X['PCA'+str(i+1)] = X_PCA[:,i]
    X = X.drop(columns=[i for i in PCA_matrix.columns])
    if Save:
        with open('./Model/Weights/pca.pkl', 'wb') as pickle_file:
            pickle.dump(pca, pickle_file)
        
    return X
        