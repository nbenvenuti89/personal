# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:19:22 2019

@author: niccolo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import scipy.stats
from datetime import datetime
from numpy import fft
import boto3



def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))            
    #indexes.sort(key = lambda i: np.absolute(f[i]))      # sort indexes by frequencies, lower -> higher
    indexes.sort(key=lambda i: np.absolute(x_freqdom[i])) # sort indexes by amplitude, lower -> higher
    indexes.reverse()    
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
    

def getCI(x,y,warning,danger):  
    n = len(x)
    m = x.mean()
    sum_errs = np.sum((x - y)**2)
    stdev = np.sqrt(1/(len(x)-2) * sum_errs)
    hw = stdev * scipy.stats.t.ppf((1 + warning) / 2, n - 1)
    hd = stdev * scipy.stats.t.ppf((1 + danger) / 2, n - 1)
    warning_high, warning_low, danger_high, danger_low = m - hw, m + hw, m - hd, m + hd
    return [hw, warning_high, warning_low], [hd, danger_high, danger_low]



def plotforecast(df, kpiName, extrapolation, warning_ci, danger_ci, confidence_interval, lower_bound_multi, upper_bound_multi, floor):
    params=np.where(df['y'][len(df)-72-1:len(df)]<np.array(extrapolation[extrapolation.size-72-1:extrapolation.size]-danger_ci[0]*lower_bound_multi),[['r'],[100]], 
                    np.where(df['y'][len(df)-72-1:len(df)]>np.array(extrapolation[extrapolation.size-72-1:extrapolation.size]+danger_ci[0]*upper_bound_multi),[['r'],[100]],
                             np.where(df['y'][len(df)-72-1:len(df)]<np.array(extrapolation[extrapolation.size-72-1:extrapolation.size]-warning_ci[0]*lower_bound_multi),[['y'],[50]],
                                      np.where(df['y'][len(df)-72-1:len(df)]>np.array(extrapolation[extrapolation.size-72-1:extrapolation.size]+warning_ci[0]*upper_bound_multi),[['y'],[50]],[['gray'],[15]]))))
    fig = plt.figure(figsize=(18.5,10.5))
    plt.subplot(211);
    plt.plot(np.arange(len(df)-72-1,len(df)), extrapolation[len(df)-72-1:len(df)], 'b', label = 'predicted')
    plt.scatter(np.arange(len(df)-72-1,len(df)), df['y'][len(df)-72-1:len(df)], s=params[1].astype(np.int64), label='real', c=np.array(params[0], dtype='<U1'))#, 'b', label = 'x', linewidth = 0.1)
    plt.xticks(np.arange(len(df)-72-1,len(df), step=6), df['ds'][np.arange(len(df)-72-1,len(df), step=6)].dt.strftime("%m/%d %H:%M"), rotation = '30')        
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.fill_between(np.arange(len(df)-72, len(df)), 
                     np.array(np.where(extrapolation[len(df)-72:len(df)]-warning_ci[0]*lower_bound_multi<floor, floor, extrapolation[len(df)-72:len(df)]-warning_ci[0]*lower_bound_multi)), 
                     np.array(extrapolation[len(df)-72:len(df)]+warning_ci[0]*upper_bound_multi), 
                     color = '#bad7df', alpha = 0.4, label = str(confidence_interval-.1)+'-'+str(confidence_interval)+'% CI')
    plt.fill_between(np.arange(len(df)-72, len(df)), 
                     np.array(np.where(extrapolation[len(df)-72:len(df)]-danger_ci[0]*lower_bound_multi<floor, floor, extrapolation[len(df)-72:len(df)]-danger_ci[0]*lower_bound_multi)), 
                     np.array(extrapolation[len(df)-72:len(df)]+danger_ci[0]*upper_bound_multi), 
                     color = '#bad7df', alpha = 0.4) ##539caf
    plt.title('KPI = '+str(kpiName)+' in the last 72 hours (Hong Kong Time)')
    plt.subplot(212);
    plt.plot(np.arange(0, len(df)), df['y'], 'b', label='real')
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.close()
    return fig



def plotforecastTest(df, kpiName, extrapolation, warning_ci, danger_ci, confidence_interval, lower_bound_multi, upper_bound_multi, floor):
    params=np.where(df['y']<np.array(extrapolation-danger_ci[0]*lower_bound_multi),[['r'],[100]], 
                    np.where(df['y']>np.array(extrapolation+danger_ci[0]*upper_bound_multi),[['r'],[100]],
                             np.where(df['y']<np.array(extrapolation-warning_ci[0]*lower_bound_multi),[['y'],[50]],
                                      np.where(df['y']>np.array(extrapolation+warning_ci[0]*upper_bound_multi),[['y'],[50]],[['gray'],[15]]))))
    fig = plt.figure(figsize=(18.5,10.5))
    plt.subplot(211);
    plt.plot(np.arange(0,len(df)), extrapolation, 'b', label = 'predicted')
    plt.scatter(np.arange(0,len(df)), df['y'], s=params[1].astype(np.int64), label='real', c=np.array(params[0], dtype='<U1'))#, 'b', label = 'x', linewidth = 0.1)
    #plt.xticks(np.arange(0,len(df), step=6), df['ds'][np.arange(0,len(df), step=6)].dt.strftime("%m/%d %H:%M"), rotation = '30')        
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.fill_between(np.arange(0, len(df)), 
                     np.array(np.where(extrapolation-warning_ci[0]*lower_bound_multi<floor, floor, extrapolation-warning_ci[0]*lower_bound_multi)), 
                     np.array(extrapolation+warning_ci[0]*upper_bound_multi), 
                     color = '#bad7df', alpha = 0.4, label = str(confidence_interval-.1)+'-'+str(confidence_interval)+'% CI')
    plt.fill_between(np.arange(0, len(df)), 
                     np.array(np.where(extrapolation-danger_ci[0]*lower_bound_multi<floor, floor, extrapolation-danger_ci[0]*lower_bound_multi)), 
                     np.array(extrapolation+danger_ci[0]*upper_bound_multi), 
                     color = '#bad7df', alpha = 0.4) ##539caf
    plt.title('KPI = '+str(kpiName)+' in the last hours (Hong Kong Time)')
    plt.subplot(212);
    plt.plot(np.arange(0, len(df)), df['y'], 'b', label='real')
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.close()
    return fig
    

def plotseries(df, kpiName, problem_if, upper_bound, lower_bound):
    params=np.where(df['y'][len(df)-72-1:] > upper_bound,[['r'],[100]], 
        np.where(df['y'][len(df)-72-1:] < lower_bound,[['r'],[100]],[['gray'],[15]]))
    fig = plt.figure(figsize=(18.5,10.5))
    plt.scatter(np.arange(len(df)-72-1, len(df)), df['y'][len(df)-72-1:len(df)], s=params[1].astype(np.int64), label='real', c=np.array(params[0], dtype='<U1'))#, 'b', label = 'x', linewidth = 0.1)
    plt.plot(np.arange(len(df)-72-1, len(df)), df['y'][len(df)-72-1:len(df)], 'b', ls='-.', label='real')
    plt.xticks(np.arange(len(df)-72-1, len(df), step=6), df['ds'][np.arange(len(df)-72-1,len(df), step=6)].dt.strftime("%m/%d %H:%M"), rotation = '30')        
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.ylim(min(min(df['y'][len(df)-72-1:len(df)]), lower_bound)*.9, max(max(df['y'][len(df)-72-1:len(df)]), upper_bound)*1.1)
    if problem_if == 'High':
        plt.fill_between(np.arange(len(df)-72-1, len(df)), upper_bound, max(max(df['y'][len(df)-72-1:len(df)]), upper_bound)*1.05, color = '#bad7df', alpha = 0.4, label = str('High Threshold'))
    elif problem_if == 'Low':
        plt.fill_between(np.arange(len(df)-72-1, len(df)), lower_bound, min(min(df['y'][len(df)-72-1:len(df)]), lower_bound)*.95, color = '#bad7df', alpha = 0.4, label = str('Low Threshold'))
    else:
        plt.fill_between(np.arange(len(df)-72-1, len(df)), lower_bound, min(min(df['y'][len(df)-72-1:len(df)]), lower_bound)*.95, color = '#bad7df', alpha = 0.4, label = str('High Threshold'))
        plt.fill_between(np.arange(len(df)-72-1, len(df)), upper_bound, max(max(df['y'][len(df)-72-1:len(df)]), upper_bound)*1.05, color = '#bad7df', alpha = 0.4, label = str('Low Threshold'))
    plt.title('KPI = '+str(kpiName)+' (Rule based) in the last 72 hours (Hong Kong Time)')
    plt.close()
    return fig   
    
  
  
def plotAlertManagament(df, hours_to_plot, kpiName, AM_upper_bound, AM_lower_bound, floor):
    params=np.where(df['y'][len(df)-72-1:len(df)] < AM_lower_bound,[['r'],[100]], 
                    np.where(df['y'][len(df)-72-1:len(df)] > AM_upper_bound,[['r'],[100]], [['gray'],[15]]))
    fig = plt.figure(figsize=(18.5,10.5))
    plt.subplot(211);
    plt.plot(np.arange(len(df)-72-1,len(df)), df['y'][len(df)-72-1:len(df)], 'gray', label = 'real line')
    plt.scatter(np.arange(len(df)-72-1,len(df)), df['y'][len(df)-72-1:len(df)], s=params[1].astype(np.int64), label='real scatter', c=np.array(params[0], dtype='<U1'))#, 'b', label = 'x', linewidth = 0.1)
    plt.xticks(np.arange(len(df)-72-1,len(df), step=6), df['ds'][np.arange(len(df)-72-1,len(df), step=6)].dt.strftime("%m/%d %H:%M"), rotation = '30')        
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.fill_between(np.arange(len(df)-72, len(df)), 
                     np.array(AM_upper_bound), 
                     np.array(max(df['y'][len(df)-72-1:len(df)])), 
                     color = '#bad7df', alpha = 0.4)
    plt.fill_between(np.arange(len(df)-72, len(df)), 
                     np.array(min(df['y'][len(df)-72-1:len(df)])), 
                     np.array(AM_lower_bound), 
                     color = '#bad7df', alpha = 0.4) ##539caf
    plt.title('KPI = '+str(kpiName)+' in the last 72 hours (Hong Kong Time)')
    plt.subplot(212);
    plt.plot(np.arange(0, len(df)), df['y'], 'b', label='real')
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.close()
    return fig


    
def forecast(df, n_predict, train_loss, test_loss, kpiName, confidence_interval, delta_warning):
    x = np.array(df['y'])
    extrapolation = fourierExtrapolation(x[:-n_predict], n_predict)
    rel_rmse_train = math.sqrt(mean_squared_error(df['y'], extrapolation))/(df['y'].mean()**int(df['y'].mean()!=0))
    rel_rmse_test = math.sqrt(mean_squared_error(df['y'].iloc[len(df)-3:len(df)], extrapolation[extrapolation.size-3:extrapolation.size]))/(df['y'].iloc[len(df)-3:len(df)].mean()**int(df['y'].iloc[len(df)-3:len(df)].mean()!=0))
    train_loss += rel_rmse_train
    test_loss  += rel_rmse_test
    warning_ci, danger_ci = getCI(x,extrapolation, confidence_interval-delta_warning, confidence_interval)
    return extrapolation, train_loss, test_loss, warning_ci, danger_ci




def uploadFileAndGetUrl(filepath, filename):
    client = boto3.client('s3',
                              aws_access_key_id="AKIAQ6RGKLFLVQAH6U4O", 
                              aws_secret_access_key="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",
                              region_name='us-west-2')
    today = datetime.now().strftime("%Y-%m-%d")
    bucket_name = 'bi.playstudios.asia'
    prefixOnS3 = 'etlTesting/blood-test/plot_cache/' + today + '/'
    keyOnS3 = prefixOnS3+filename
    client.upload_file(filepath+filename, bucket_name, keyOnS3)
    url = client.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket_name,
            'Key': keyOnS3
        },
        ExpiresIn=63072000 #604800
    )

    return url