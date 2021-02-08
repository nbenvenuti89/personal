# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:29:40 2021

@author: niccolo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from Functions_v5 import fourierExtrapolation, getCI


def GetSeries(df_global, history, params):
    # Get the Series of the Kpi
    try:
        df1 = df_global.loc[len(df_global)-337-params['num_hours']:len(df_global)-2,["LOCAL_HOUR",params['kpiCode']]]
    except:
        df1 = df_global.loc[:,["LOCAL_HOUR",params['kpiCode']]]
    df1.columns = ['ds','y']
    df1 = df1.reset_index(drop=True)
    # Handling Missing
    if df1.y.isnull().sum()>0:
        if params['is_null_zero']: df1['y'].fillna(0, inplace=True)
        else:
            df1['y'] = df1['y'].interpolate()
            df1 = df1[pd.notnull(df1.y)]
    
    #------------------------------------------------#
    # Replacing Previous Single-Alert with Prediction 
    #------------------------------------------------#   
    if params['rule_based']!='True':
        check = history[(history.KPI_CODE==params['kpiCode']) & (history.RANK == 1)]
        if any(check.SINGLE_ALERT == 'True'):
            replacements = check[check.SINGLE_ALERT == 'True'][['TIMESTAMP','PREDICTIONS']]
            df2 = df1.merge(replacements, how='left', left_on='ds', right_on='TIMESTAMP')
            df2.y = np.where(df2.PREDICTIONS.isna(), df2.y, df2.PREDICTIONS)
            df2.drop(['TIMESTAMP','PREDICTIONS'], axis=1, inplace=True)
        else:
            df2 = df1
    return df1, df2          
            
            

def Forecast(df, params):
    if params['rule_based']=='True':
        extrapolation, rel_rmse_train, rel_rmse_test, warning_ci, danger_ci = None, None, None, None, None
    else:
        extrapolation = fourierExtrapolation(np.array(df['y'])[:-params['num_hours']], params['num_hours'])
        rel_rmse_train = math.sqrt(mean_squared_error(df['y'], extrapolation))/(df['y'].mean()**int(df['y'].mean()!=0))
        rel_rmse_test = math.sqrt(mean_squared_error(df['y'].iloc[len(df)-3:len(df)], extrapolation[extrapolation.size-3:extrapolation.size]))/(df['y'].iloc[len(df)-3:len(df)].mean()**int(df['y'].iloc[len(df)-3:len(df)].mean()!=0))
        warning_ci, danger_ci = getCI(np.array(df['y']),extrapolation, params['confidence_interval']-params['delta_warning'], params['confidence_interval'])
    return extrapolation, rel_rmse_train, rel_rmse_test, warning_ci, danger_ci
    




def Get_Evaluation(real, params, thresholds, worksheet):
    #### Fourier Based Logic
    if params['rule_based'] == 'False':
        if params['problem_if'] == 'High':
            is_warning = np.where(real > thresholds['warning_upper'], 'True', 'False')
            if all(real > thresholds['danger_upper']):
                is_danger = 'True'
            else:
                is_danger = 'False'
        elif params['problem_if'] == 'Low':
            is_warning = np.where(real < thresholds['warning_lower'], 'True', 'False')
            if all(real < thresholds['danger_lower']):
                is_danger = 'True'
            else:
                is_danger = 'False'
        elif params['problem_if'] == 'Both':
            is_warning = np.where((real < thresholds['warning_lower']) | (real > thresholds['warning_upper']), 'True', 'False')
            if all(real < thresholds['danger_lower']) or all(real > thresholds['danger_upper']):
                is_danger = 'True'
            else:
                is_danger = 'False'
        OTA = np.where((real < thresholds['danger_lower']) | (real > thresholds['danger_upper']), 'True', 'False')
    #### Rule Based Logic
    elif params['rule_based'] == 'True':
        if params['problem_if'] == 'High':
            if all(real > params['upper_bound']):
                is_danger = 'True'
            else:
                is_danger = 'False'
        elif params['problem_if'] == 'Low': 
            if all(real < params['lower_bound']):
                is_danger = 'True'
            else:
                is_danger = 'False'
        elif params['problem_if'] == 'Both':
            if all(real > params['upper_bound']) or all(real < params['lower_bound']):        
                is_danger = 'True'
            else:
                is_danger = 'False'
        OTA = np.where((real < params['lower_bound']) | (real > params['upper_bound']), 'True', 'False')
    #### Alert Management
    elif params['rule_based'] == 'Alert_management':
        if params['problem_if'] == 'High':
            is_warning = np.where(real > thresholds['warning_upper'], 'True', 'False')
            if all(real > thresholds['danger_upper']):
                is_danger = 'True'
            else:
                is_danger = 'False'
            if any(real > thresholds['AM_upper_bound']):
                params['hours_to_plot' ] = params['tot_hours']
            else:
                if params['hours_to_plot' ] != 0:
                    params['hours_to_plot' ] -= 1
            worksheet.update_value('O'+str(params['source_index']), params['hours_to_plot' ])
        elif params['problem_if'] == 'Low':
            is_warning = np.where(real < thresholds['warning_lower'], 'True', 'False')
            if all(real < thresholds['danger_lower']):
                is_danger = 'True'
            else:
                is_danger = 'False'
            if any(real < thresholds['AM_lower_bound']):
                params['hours_to_plot'] = params['tot_hours']
            else:
                if params['hours_to_plot'] != 0:
                    params['hours_to_plot'] -= 1
            worksheet.update_value('O'+str(params['source_index']), params['hours_to_plot'])            
        elif params['problem_if'] == 'Both':
            is_warning = np.where((real < thresholds['warning_lower']) | (real > thresholds['warning_upper']), 'True', 'False')
            if all(real < thresholds['danger_lower']) or all(real > thresholds['danger_upper']):
                is_danger = 'True'
            else:
                is_danger = 'False'
            if any(real < thresholds['AM_lower_bound']) or any(real > thresholds['AM_upper_bound']):
                params['hours_to_plot'] = params['tot_hours']
            else:
                if params['hours_to_plot'] != 0:
                    params['hours_to_plot'] -= 1
            worksheet.update_value('O'+str(params['source_index']), params['hours_to_plot'])
        
        # condition to go back to normal
        if is_warning == 'False':
            params['consecutive_alerts'] -= 1
            worksheet.update_value('W'+str(params['source_index']), params['consecutive_alerts'])
            if params['consecutive_alerts'] == 0:
                worksheet.update_value('G'+str(params['source_index']), params['original_rule'])                   
                worksheet.update_value('O'+str(params['source_index']), params['tot_hours'])
                worksheet.update_value('Y'+str(params['source_index']), 'DEV')
        else:
            params['consecutive_alerts'] = 3
            worksheet.update_value('W'+str(params['source_index']), params['consecutive_alerts'])
        OTA = np.where((real < thresholds['danger_lower']) | (real > thresholds['danger_upper']), 'True', 'False')   
    
    # Condition to switch back to Prod Channel
    if (params['slack_channel'] == 'DEV') and (params['dev2prod']>0):
        if is_danger == 'False':
            params['dev2prod'] -= 1
        else:
            params['dev2prod'] = params['tot_hours']
        worksheet.update_value('Z'+str(params['source_index']), params['dev2prod'])
        
        return is_warning, is_danger, OTA