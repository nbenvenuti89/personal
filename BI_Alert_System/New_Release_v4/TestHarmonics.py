# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:45:07 2020

@author: niccolo
"""


####################################################################################################################
#-------------------------------------------------- MAGNUM BI VERSION 4.0 -----------------------------------------#
####################################################################################################################


#-------------------------------------------------------- PHASE 1 -------------------------------------------------#
#----------------------------------------------------- STARTING JOB -----------------------------------------------#


#################################
# Import Libraries and Functions
#################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
import pygsheets
import pytz
from datetime import datetime

# Get Current Directory
os.chdir("C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4")

# Import Functions
import Functions_v4 as Functions
import DB_Connection 


########################
# Parameters
########################

Test  = False # Uses personal slack channels and saves messages locally  
Plots_Folder = 'C://Users/niccolo/Desktop/BI_Alert_System/Graphic_Analysis/TestingHarmonics'

########################
# Dictionaries
########################

Database = {True:'mvmmart.public.BI_ALERT_TEST', 
            False:'mvmmart.public.BI_ALERT'}


#-------------------------------------------------------- PHASE 2 -------------------------------------------------#
#------------------------------------------------------- GET DATA -------------------------------------------------#

#############################################
# Reading tables DB and KPIs
# Setting Kpis to forecast
#############################################

sql_globalKpis = []
globalKpis = []


# import Kpi Table
gc = pygsheets.authorize(service_file='C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Python-AUTH-dfd0f213ea4e.json')
spreadsheet_object = gc.open_by_key("1mM7zvhlQ9uspKhZd3TfOtDHzWGM2FJpb3OlrlmsIS1s")
worksheet = spreadsheet_object.worksheet(0)
raw_kpi_table = pd.DataFrame(worksheet.get_all_values(include_tailing_empty = False, include_tailing_empty_rows = False))
raw_kpi_table.rename(columns=raw_kpi_table.iloc[0], inplace = True)
raw_kpi_table = raw_kpi_table.iloc[1:,:]
#raw_kpi_table = raw_kpi_table.replace('', np.NaN, regex=False)
for row in range(1,len(raw_kpi_table)+1):
    if raw_kpi_table['if_invalid'][row]=='':
        item = {}
        item['globalSql'] = raw_kpi_table['sql'][row]
        globalKpis.append(item['globalSql'])
        sql_globalKpis = ','.join(globalKpis) # Generate the SQL that gets all KPI data at just one query. group by platform, location, appversion

# read table KPIs 
raw_kpi_table2 = raw_kpi_table[raw_kpi_table["if_invalid"]==''].reset_index().replace('NA', np.NaN)



# Parameter definition
kpiCodes = raw_kpi_table2["kpi_code"]
kpiNames = raw_kpi_table2["kpi_name"]
kpiNumHours = raw_kpi_table2["num_hours"]
kpiHoursToPlot = raw_kpi_table2["hours_to_plot"]
kpiRuleBased = raw_kpi_table2["rule_based"]
kpiUpperBound = raw_kpi_table2["upper_bound"]
kpiLowerBound = raw_kpi_table2["lower_bound"]
kpiFloor = raw_kpi_table2["floor"]
kpiProblemIf = raw_kpi_table2["problem_if"].str.lower()
kpiConfidenceInterval = raw_kpi_table2["confidence_level"]
kpiLowerBoundMulti  = raw_kpi_table2["lower_CI_multiplier"] # Multipliers on confidence interval lower bound (raising this parameters will make the Alert less strict)
kpiUpperBoundMulti  = raw_kpi_table2["upper_CI_multiplier"] # Multipliers on confidence interval lower bound (raising this parameters will make the Alert less strict)
kpiDeltaWarning  = raw_kpi_table2["delta_CI_warning"] # Difference in Confidence between Warning and Danger Alerts (the highest the smallest warning Confindence Interval will be)
kpiTableau = raw_kpi_table2["tableau_url"]
kpiIsNullZero = raw_kpi_table2["is_null_zero"]
kpi_index = raw_kpi_table2["index"]
kpiConsecutiveAlerts = raw_kpi_table2["consecutive_alerts"]
kpiTotHourstoPlot = raw_kpi_table2["tot_hours_to_plot"]
kpiChannel = raw_kpi_table2["channel"]
kpiDev2Prod = raw_kpi_table2["dev2prod"]
kpiRestart = raw_kpi_table2["restart"]
kpiTagWynn = raw_kpi_table2["tag_wynn"]
kpiTagJames = raw_kpi_table2["tag_james"]
kpiTagNico = raw_kpi_table2["tag_nico"]
kpiTagKris = raw_kpi_table2["tag_kris"]
    
   
# Aurora Connection for Actual Values
ctx_Aurora = DB_Connection.AURORA_connection()
# Return the past 10 days' data, group by platform, location, appversion
df_global = pd.read_sql('''
    select CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') as LOCAL_HOUR, ''' + sql_globalKpis + ''' from nrtreporting.mx_magnum_bi 
    where date >= date_sub(CURRENT_DATE(), INTERVAL 15 DAY) 
    group by 1 order by 1
    '''  , ctx_Aurora)
ctx_Aurora.close()


# MVMMART Connection for Past Predictions values
ctx_Asia = DB_Connection.SN_connection()
# Return the past 10 days' data, group by platform, location, appversion
history = pd.read_sql('''
select *,
case when real_values>alert_upper_bound and rank = 1 then alert_upper_bound
when real_values < alert_lower_bound and rank = 1 then alert_lower_bound else real_values end as Replacements
from '''+str(Database[Test])+''' 
where TIMESTAMP between dateadd('hour', -333, (select max(TIMESTAMP) from '''+str(Database[Test])+'''))
and dateadd('hour', -1, (select max(TIMESTAMP) from '''+str(Database[Test])+''')) 
order by TIMESTAMP, KPI, RANK
'''  , ctx_Asia)
ctx_Asia.close()




#-------------------------------------------------------- PHASE 3 -------------------------------------------------#
#------------------------------------------------------- ALGORITHM ------------------------------------------------#


#############################################
# For loop 
# every Kpi available
#############################################

cols = ['KPI_Code','KPI','Date','Real_Values','Predictions','Warning_Lower_Bound','Warning_Upper_Bound','Alert_Lower_Bound','Alert_Upper_Bound','RMSE','RMSE_Test','Is_Warning','Is_Alert','Rule_Based','OTA','Rank'] 
forecast= pd.DataFrame(columns=cols)

'''k = 15
for j, e in enumerate(kpiCodes[k:k+1]):
    j=k
    print('j = ' + str(j))
    print('e = ' + str(e))
'''    


for j, e in enumerate(kpiCodes):
    print('j = ' + str(j))
    print('e = ' + str(e))

    # Getting Variables  
    kpiCode=str(e)
    kpiName=str(kpiNames[j])
    source_index = int(kpi_index[j])
    consecutive_alerts = int(kpiConsecutiveAlerts[j])
    tot_hours = int(kpiTotHourstoPlot[j])
    rule_based=kpiRuleBased[j].capitalize()
    num_hours=int(kpiNumHours[j])
    confidence_interval=float(kpiConfidenceInterval[j])
    lower_bound_multi = float(kpiLowerBoundMulti[j])
    upper_bound_multi = float(kpiUpperBoundMulti[j])
    delta_warning = float(kpiDeltaWarning[j])
    floor=float(kpiFloor[j])
    upper_bound=float(kpiUpperBound[j])
    lower_bound=float(kpiLowerBound[j])
    problem_if=kpiProblemIf[j].capitalize()
    hours_to_plot=int(kpiHoursToPlot[j])
    slack_channel = str(kpiChannel[j])
    dev2prod = int(kpiDev2Prod[j])
    restart=kpiRestart[j].capitalize()
    tableau=kpiTableau[j]
    is_null_zero=kpiIsNullZero[j]
    tags=[kpiTagWynn[j],kpiTagJames[j],kpiTagNico[j],kpiTagKris[j]]   
    
    
    #########################
    # Data Processing by KPI
    #########################
    
    if restart == 'True':
        worksheet.update_value('G'+str(source_index), 'FALSE') 
        worksheet.update_value('O'+str(source_index), tot_hours) 
        worksheet.update_value('W'+str(source_index), 0) 
        worksheet.update_value('Y'+str(source_index), 'PROD') 
        worksheet.update_value('Z'+str(source_index), tot_hours) 
        worksheet.update_value('AA'+str(source_index), 'FALSE') 
        consecutive_alerts = 0
        hours_to_plot = tot_hours
        slack_channel = 'PROD'
        rule_based = 'False'
        dev2prod = tot_hours
        
        
    # Filtering table by single KPI
    try:
        df1 = df_global.loc[len(df_global)-337-num_hours:len(df_global)-2,["LOCAL_HOUR",kpiCode]]
    except:
        df1 = df_global.loc[:,["LOCAL_HOUR",kpiCode]]
    df1.columns = ['ds','y']
    df1 = df1.reset_index(drop=True)
    
    # Handling Missing
    if df1.y.isnull().sum()>0:
        if is_null_zero:
            df1['y'].fillna(0, inplace=True)
        else:
            df1['y'] = df1['y'].interpolate()
            df1 = df1[pd.notnull(df1.y)]
                      
                
    
    #########################
    # Start Predicting
    #########################
       
    try:    
        start_date = np.array(df1['ds'][len(df1)-num_hours:])
        real = np.array(df1.y[len(df1)-num_hours:len(df1)])       
        
        if rule_based=='False': 
            print('Rule Based = False')
            ############################################################
            # Alert based on Fourier-Algorithm
            # Prediction Based Alerts
            ############################################################
        
            #------------------------------------------------#
            # Replacing Previous Single-Alert with Prediction 
            #------------------------------------------------#   
            check = history[(history.KPI_CODE==kpiCode) & (history.RANK == 1)]
            if any(check.SINGLE_ALERT == 'True'):
                replacements = check[check.SINGLE_ALERT == 'True'][['TIMESTAMP','PREDICTIONS']]
                df2 = df1.merge(replacements, how='left', left_on='ds', right_on='TIMESTAMP')
                df2.y = np.where(df2.PREDICTIONS.isna(), df2.y, df2.PREDICTIONS)
                df2.drop(['TIMESTAMP','PREDICTIONS'], axis=1, inplace=True)
            else:
                df2 = df1
                
                
            #----------------------------------------------#
            # Fourier Trasformation and prediction algorithm
            #----------------------------------------------#
            train_loss, test_loss = 0, 0
            prediction, train_loss, test_loss, warning_ci, danger_ci = Functions.forecast(df2, num_hours, train_loss, test_loss, kpiName, confidence_interval, delta_warning)
            pred = np.array(prediction[len(prediction)-num_hours:len(prediction)])
            pred = np.where(pred<floor, floor, pred)
            
            #---------------------------------------------------------------------#
            # Tresholds (only floor for now, to add cap in the future if necessary)
            #---------------------------------------------------------------------#
            warning_upper = np.where((pred+warning_ci[0]*upper_bound_multi)<floor,floor,(pred+warning_ci[0]*upper_bound_multi))
            warning_lower = np.where((pred-warning_ci[0]*lower_bound_multi)<floor,floor,(pred-warning_ci[0]*lower_bound_multi))
            danger_upper  = np.where((pred+danger_ci[0]*upper_bound_multi)<floor,floor,(pred+danger_ci[0]*upper_bound_multi))
            danger_lower  = np.where((pred-danger_ci[0]*lower_bound_multi)<floor,floor,(pred-danger_ci[0]*lower_bound_multi))
                 
        
            #--------------------------------#            
            # Append info in the results table
            #--------------------------------#
            for nrows in range(num_hours):
                forecast = forecast.append(pd.DataFrame({'KPI_Code': [kpiCode], 'KPI': [kpiName], 'Date': [pd.to_datetime(str(start_date[nrows]))], 'Real_Values':[real[nrows]], 'Predictions':[pred[nrows]], 'Warning_Lower_Bound' : [warning_lower[nrows]],'Warning_Upper_Bound' : [warning_upper[nrows]],'Alert_Lower_Bound' : [danger_lower[nrows]],'Alert_Upper_Bound' : [danger_upper[nrows]]}), sort=True)
            
            #----------------------------#
            # Alert Plot and Slack Message
            #----------------------------#
            fig = Functions.plotforecast(df1, kpiName, prediction, warning_ci, danger_ci,  confidence_interval, lower_bound_multi, upper_bound_multi, floor)
            fig
            fig.savefig(str(Plots_Folder)+'\\Alert of '+str(kpiName)+'_'+pd.to_datetime(str(start_date[0])).strftime("%d_%m %H")+'.png')
        
        elif rule_based == 'True': 
            print('Rule Based = True')
            ############################################################
            # Alert based on defined rules
            # From Google Sheet
            ############################################################
            
            prediction, warning_ci, danger_ci = None, None, None
                
            #--------------------------------#            
            # Append info in the results table
            #--------------------------------#
            for nrows in range(num_hours):
                forecast = forecast.append(pd.DataFrame({'KPI_Code': [kpiCode], 'KPI': [kpiName], 'Date': [pd.to_datetime(str(start_date[nrows]))], 'Real_Values':[real[nrows]], 'Predictions':'NULL', 'Warning_Lower_Bound' : 'NULL','Warning_Upper_Bound' : 'NULL','Alert_Lower_Bound' : [lower_bound],'Alert_Upper_Bound' : [upper_bound]}), sort=True)
        
            #----------------------------#
            # Alert Plot and Slack Message
            #----------------------------#
            fig = Functions.plotseries(df1, kpiName, problem_if, upper_bound, lower_bound)
            fig
            fig.savefig(str(Plots_Folder)+'\\Alert of '+str(kpiName)+'_'+pd.to_datetime(str(start_date[0])).strftime("%d_%m %H")+'.png')

    except:
        print('Error for '+str(kpiName))
        next
    
    
    
    
#-------------------------------------------------------- PHASE 4 -------------------------------------------------#
#------------------------------------------------------- SAVE DATA ------------------------------------------------#

# resetting index for the upload
forecast = forecast.reset_index(drop=True)
forecast.to_csv('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\\BI_Alert_System_'+str(datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H"))+'.csv', index=False)           
