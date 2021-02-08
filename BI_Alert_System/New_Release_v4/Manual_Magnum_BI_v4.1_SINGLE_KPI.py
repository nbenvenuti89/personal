# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:07:08 2020

@author: niccolo
"""


####################################################################################################################
#-------------------------------------------------- MAGNUM BI VERSION 4.1 -----------------------------------------#
####################################################################################################################


#-------------------------------------------------------- PHASE 1 -------------------------------------------------#
#----------------------------------------------------- STARTING JOB -----------------------------------------------#


#################################
# Import Libraries and Functions
#################################

import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import pygsheets
import pytz
from datetime import datetime, timedelta

# Get Current Directory
os.chdir("C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4")

# Import Functions
import Functions_v4 as Functions
import DB_Connection 
import Slack_Sender_v4 as Slack_Sender


########################
# Parameters
########################

Local = False # If local does not send Slack Messages / Save results in DB 
Test  = True # Uses personal slack channels and saves messages locally  
e = str(sys.argv[1])
j = str(sys.argv[2])
APP = 'BLENDED_MVM_MX' # do delete


########################
# Dictionaries
########################

Database = {True:'mvmmart.public.BI_ALERT_TEST', 
            False:'mvmmart.public.BI_ALERT'}

if Test:
    url = {'PROD': 'https://hooks.slack.com/services/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', # bi-alert_new
           'DEV' : 'https://hooks.slack.com/services/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', # bi-alert_new
           'PERSONAL' : 'https://hooks.slack.com/services/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', # personal-Slackbot 
           'BI_ALERT_NEW': 'https://hooks.slack.com/services/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'} # bi-alert_new
else:
    url = {'PROD': 'https://hooks.slack.com/services/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', # mx-magnum-bi
           'DEV' : 'https://hooks.slack.com/services/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'} # mx-blue-steel




#-------------------------------------------------------- PHASE 2 -------------------------------------------------#
#------------------------------------------------------- GET DATA -------------------------------------------------#

#############################################
# Reading tables DB and KPIs
# Setting Kpis to forecast
#############################################
days = pd.date_range('2020-01-27 09:50:00', periods=round((datetime.now()-datetime(2020,1,27,9,50,0)).total_seconds()/3600)+1, freq='H').tolist()
days = days[:-1]
for i, current_date in enumerate(days):
    print(current_date)
    if max(days)<datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')-timedelta(hours=1):
        days.append(datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:50:00'),'%Y-%m-%d %H:%M:%S'))
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
    if APP=='MVM':
        df_global = pd.read_sql('''
        select CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') as LOCAL_HOUR, ''' + sql_globalKpis + ''' from nrtreporting.mx_magnum_bi 
        where CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') >= date_sub("'''+ str(current_date)+'''", INTERVAL 337 HOUR) 
        and CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') <= "'''+ str(current_date)+'''" and version not like '3.%'
        group by 1 order by 1
        '''  , ctx_Aurora)
    elif APP=='MX':
        df_global = pd.read_sql('''
        select CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') as LOCAL_HOUR, ''' + sql_globalKpis + ''' from nrtreporting.mx_magnum_bi 
        where CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') >= date_sub("'''+ str(current_date)+'''", INTERVAL 337 HOUR) 
        and CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') <= "'''+ str(current_date)+'''" and version like '3.%'
        group by 1 order by 1
        '''  , ctx_Aurora)
    else:
        df_global = pd.read_sql('''
        select CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') as LOCAL_HOUR, ''' + sql_globalKpis + ''' from nrtreporting.mx_magnum_bi 
        where CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') >= date_sub("'''+ str(current_date)+'''", INTERVAL 337 HOUR) 
        and CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') <= "'''+ str(current_date)+'''" 
        group by 1 order by 1
        '''  , ctx_Aurora)
    ctx_Aurora.close()


    # MVMMART Connection for Past Predictions values
    ctx_Asia = DB_Connection.SN_connection()
    # Return the past 10 days' data, group by platform, location, appversion
    history = pd.read_sql('''
    select *,
    row_number() over (partition by RUN_TIMESTAMP, KPI order by TIMESTAMP) as rank from '''+str(Database[Test])+''' 
    where TIMESTAMP between dateadd('hour', -333, (select max(TIMESTAMP) from '''+str(Database[Test])+'''))
    and dateadd('hour', -1, (select max(TIMESTAMP) from '''+str(Database[Test])+''')) and APP='''+"'"+str(APP)+"'"+'''  
    order by TIMESTAMP, KPI, RANK
    '''  , ctx_Asia)
    ctx_Asia.close()




    #-------------------------------------------------------- PHASE 3 -------------------------------------------------#
    #------------------------------------------------------- ALGORITHM ------------------------------------------------#


    #############################################
    # For loop 
    # every Kpi available
    #############################################

    cols = ['KPI_Code','KPI','Date','Real_Values','Predictions','Warning_Lower_Bound','Warning_Upper_Bound','Alert_Lower_Bound','Alert_Upper_Bound','RMSE','RMSE_Test','Is_Warning','Is_Alert','Rule_Based','OTA'] 
    forecast= pd.DataFrame(columns=cols)
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
        
        # Change of Alerts logic in case of more than 3 consecutive alerts
        if ((consecutive_alerts >= 3) and (rule_based == 'False')):
            worksheet.update_value('G'+str(source_index), 'ALERT_MANAGEMENT') # Update Value on Google Sheet
            worksheet.update_value('Z'+str(source_index), tot_hours) 
            #worksheet.update_value('W'+str(source_index), 0)
            print('Switch to Alert Management')
            rule_based = 'Alert_management'
            
        
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
                
            '''
            #----------------------------------------#
            # Replacing Previous Alert with Prediction 
            #----------------------------------------#   
            check = history[(history.KPI_CODE==kpiCode) & (history.RANK == 1)]
            if any(check.IS_ALERT == 'True'):
                replacements = check[check.IS_WARNING == 'True'][['TIMESTAMP','PREDICTIONS']]
                df2 = df1.merge(replacements, how='left', left_on='ds', right_on='TIMESTAMP')
                df2.y = np.where(df2.PREDICTIONS.isna(), df2.y, df2.PREDICTIONS)
                df2.drop(['TIMESTAMP','PREDICTIONS'], axis=1, inplace=True)
            else:
                df2 = df1
            '''
            
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
            
            #----------------------------------#
            # Handling Warning and Danger Alerts
            #----------------------------------#
            if problem_if == 'High':
                is_warning = np.where(real > warning_upper, 'True', 'False')
                if all(real > danger_upper):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            elif problem_if == 'Low':
                is_warning = np.where(real < warning_lower, 'True', 'False')
                if all(real < danger_lower):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            elif problem_if == 'Both':
                is_warning = np.where((real < warning_lower) | (real > warning_upper), 'True', 'False')
                if all(real < danger_lower) or all(real > danger_upper):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            
            # Handling Outliers in the distribution - OTA (one time alerts)
            OTA = np.where((real < danger_lower) | (real > danger_upper), 'True', 'False')
            
            # Condition to switch back to Prod Channel
            if (slack_channel == 'DEV') and (dev2prod>0):
                if is_danger == 'False':
                    dev2prod -= 1
                else:
                    dev2prod = tot_hours
                worksheet.update_value('Z'+str(source_index), dev2prod)
            
            #--------------------------------#            
            # Append info in the results table
            #--------------------------------#
            for nrows in range(num_hours):
                forecast = forecast.append(pd.DataFrame({'KPI_Code': [kpiCode], 'KPI': [kpiName], 'Date': [pd.to_datetime(str(start_date[nrows]))], 'Real_Values':[real[nrows]], 'Predictions':[pred[nrows]], 'Warning_Lower_Bound' : [warning_lower[nrows]],'Warning_Upper_Bound' : [warning_upper[nrows]],'Alert_Lower_Bound' : [danger_lower[nrows]],'Alert_Upper_Bound' : [danger_upper[nrows]], 'RMSE':[train_loss], 'RMSE_Test':[test_loss], 'Is_Warning':[is_warning[nrows]], 'Is_Alert':[is_danger], 'Rule_Based':[rule_based],'OTA':[OTA[nrows]]}), sort=True)
                if nrows == 0:
                    realvalue_prediction = 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+' | Expected: '+str(f'{round(pred[nrows], 2):.2f}')+'\n'
                else:
                    realvalue_prediction += 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+' | Expected: '+str(f'{round(pred[nrows], 2):.2f}')+'\n'

            #----------------------------#
            # Alert Plot and Slack Message
            #----------------------------#
            if is_danger == 'True': 
                worksheet.update_value('W'+str(source_index), consecutive_alerts+1) # Update Value on Google Sheet
                fig = Functions.plotforecast(df1, kpiName, prediction, warning_ci, danger_ci,  confidence_interval, lower_bound_multi, upper_bound_multi, floor)
                if Local:
                    plt.savefig('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\Alert of '+str(int(confidence_interval*100))+'% '+pd.to_datetime(str(start_date[0])).strftime("%d_%m %H")+'.png')
                    plt.close()
                else:
                    Slack_Sender.Message_out_Predictions(kpiName, url[slack_channel], fig, tags, realvalue_prediction)
            else:
                worksheet.update_value('W'+str(source_index), 0)

            
            #-----------------------------------------#
            # Slack Message Everything's Back to Normal
            #-----------------------------------------#
            if dev2prod == 0:
                worksheet.update_value('Z'+str(source_index), tot_hours) 
                worksheet.update_value('Y'+str(source_index), 'PROD')
                slack_channel = 'PROD'
                fig = Functions.plotforecast(df1, kpiName, prediction, warning_ci, danger_ci,  confidence_interval, lower_bound_multi, upper_bound_multi, floor)
                Slack_Sender.Message_out_Back2Normal(kpiName, url[slack_channel], fig, tags, realvalue_prediction)

        
        elif rule_based == 'True': 
            print('Rule Based = True')
            ############################################################
            # Alert based on defined rules
            # From Google Sheet
            ############################################################
            
            prediction, warning_ci, danger_ci = None, None, None
            real = np.array(df1.y[len(df1)-num_hours:len(df1)])
            
            #----------------------------------#
            # Handling Warning and Danger Alerts
            #----------------------------------#
            if problem_if == 'High':
                if all(real > upper_bound):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            elif problem_if == 'Low': 
                if all(real < lower_bound):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            elif problem_if == 'Both':
                if all(real > upper_bound) or all(real < lower_bound):        
                    is_danger = 'True'
                else:
                    is_danger = 'False'
            
            # Handling Outliers in the distribution - OTA (one time alerts)
#                OTA = np.where((real < lower_bound) | (real > upper_bound), 'True', 'False')
            
            
            #--------------------------------#            
            # Append info in the results table
            #--------------------------------#
            for nrows in range(num_hours):
                forecast = forecast.append(pd.DataFrame({'KPI_Code': [kpiCode], 'KPI': [kpiName], 'Date': [pd.to_datetime(str(start_date[nrows]))], 'Real_Values':[real[nrows]], 'Predictions':'NULL', 'Warning_Lower_Bound' : 'NULL','Warning_Upper_Bound' : 'NULL','Alert_Lower_Bound' : [lower_bound],'Alert_Upper_Bound' : [upper_bound], 'RMSE':'NULL', 'RMSE_Test':'NULL', 'Is_Warning':['False'], 'Is_Alert':[is_danger], 'Rule_Based':[rule_based],'OTA':[OTA[nrows]]}), sort=True)
                if nrows == 0:
                    realvalue_prediction = 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+'\n'
                    boundaries = 'Lower Bound: '+str(f'{round(lower_bound, 2):.2f}')+'\n Upper Bound: '+str(f'{round(upper_bound, 2):.2f}')
                else:
                    realvalue_prediction += 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+'\n'

            #----------------------------#
            # Alert Plot and Slack Message
            #----------------------------#
            if is_danger == 'True':  
                worksheet.update_value('W'+str(source_index), consecutive_alerts+1) # Update Value on Google Sheet
                if Local:
                    fig = Functions.plotseries(df1, kpiName, problem_if, upper_bound, lower_bound)
                    plt.savefig('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\Alert of '+str(kpiName)+'_'+pd.to_datetime(str(start_date[0])).strftime("%d_%m %H")+'.png')
                    plt.close()
                else:
                    fig = Functions.plotseries(df1, kpiName, problem_if, upper_bound, lower_bound)
                    Slack_Sender.Message_out_Rule_Based(kpiName, url[slack_channel], fig, tags, boundaries, realvalue_prediction)
            else:
                worksheet.update_value('W'+str(source_index), 0)
        
        
        
        
        
        elif rule_based == 'Alert_management':
            print('Rule Based = Alert Management')
            ####################################################################
            # Alert Management - 
            # Plot whatever happens for "hours_to_plot" times
            ####################################################################
            
            #----------------------------------------#
            # Replacing Previous Alert with Prediction 
            #----------------------------------------#  
            AM_upper_bound = max(df1.y.iloc[:-num_hours])
            AM_lower_bound = min(df1.y.iloc[:-num_hours])
            check = history[(history.KPI_CODE==kpiCode) & (history.RANK == 1)]
            replacements = check[check.IS_WARNING == 'True'][['TIMESTAMP','PREDICTIONS']]
            df2 = df1.merge(replacements, how='left', left_on='ds', right_on='TIMESTAMP')
            df2.y = np.where(df2.PREDICTIONS.isna(), df2.y, df2.PREDICTIONS)
            df2.drop(['TIMESTAMP','PREDICTIONS'], axis=1, inplace=True)
    
            #----------------------------------------------#
            # Fourier Trasformation and prediction algorithm
            #----------------------------------------------#
            train_loss, test_loss = 0, 0
            prediction, train_loss, test_loss, warning_ci, danger_ci = Functions.forecast(df2, num_hours, train_loss, test_loss, kpiName, confidence_interval, delta_warning)
            pred = np.array(prediction[len(prediction)-num_hours:len(prediction)])
            pred = np.where(pred<floor, floor, pred)
            
            #----------#
            # Tresholds
            #----------#
            warning_upper = np.where((pred+warning_ci[0]*upper_bound_multi)<floor,floor,(pred+warning_ci[0]*upper_bound_multi))
            warning_lower = np.where((pred-warning_ci[0]*lower_bound_multi)<floor,floor,(pred-warning_ci[0]*lower_bound_multi))
            danger_upper = np.where((pred+danger_ci[0]*upper_bound_multi)<floor,floor,(pred+danger_ci[0]*upper_bound_multi))
            danger_lower = np.where((pred-danger_ci[0]*lower_bound_multi)<floor,floor,(pred-danger_ci[0]*lower_bound_multi))
            
            #worksheet.update_value('H'+str(source_index), upper_bound)
            #worksheet.update_value('I'+str(source_index), lower_bound)       
            
            #----------------------------------#
            # Handling Warning and Danger Alerts
            #----------------------------------#
            if problem_if == 'High':
                is_warning = np.where(real > warning_upper, 'True', 'False')
                if all(real > danger_upper):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
                if any(real > AM_upper_bound):
                    hours_to_plot = tot_hours
                else:
                    if hours_to_plot != 0:
                        hours_to_plot -= 1
                worksheet.update_value('O'+str(source_index), hours_to_plot)
                # condition to go back to normal
                if all(real < danger_upper): 
                    consecutive_alerts -= 1
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
                    if consecutive_alerts == 0:
                        worksheet.update_value('G'+str(source_index), 'FALSE')                   
                        worksheet.update_value('O'+str(source_index), tot_hours)
                        worksheet.update_value('Y'+str(source_index), 'DEV')
                else:
                    consecutive_alerts = 3
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
            elif problem_if == 'Low':
                is_warning = np.where(real < warning_lower, 'True', 'False')
                if all(real < danger_lower):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
                if any(real < AM_lower_bound):
                    hours_to_plot = tot_hours
                else:
                    if hours_to_plot != 0:
                        hours_to_plot -= 1
                worksheet.update_value('O'+str(source_index), hours_to_plot)
                # condition to go back to normal
                if all(real > danger_lower): 
                    consecutive_alerts -= 1
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
                    if consecutive_alerts == 0:
                        worksheet.update_value('G'+str(source_index), 'FALSE')                   
                        worksheet.update_value('O'+str(source_index), tot_hours)
                        worksheet.update_value('Y'+str(source_index), 'DEV')
                else:
                    consecutive_alerts = 3
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
            elif problem_if == 'Both':
                is_warning = np.where((real < warning_lower) | (real > warning_upper), 'True', 'False')
                if all(real < danger_lower) or all(real > danger_upper):
                    is_danger = 'True'
                else:
                    is_danger = 'False'
                if any(real < AM_lower_bound) or any(real > AM_upper_bound):
                    hours_to_plot = tot_hours
                else:
                    if hours_to_plot != 0:
                        hours_to_plot -= 1
                worksheet.update_value('O'+str(source_index), hours_to_plot)
                # condition to go back to normal
                if all(real > warning_lower) and all(real < warning_upper):
                    consecutive_alerts -= 1
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
                    if consecutive_alerts == 0:
                        worksheet.update_value('G'+str(source_index), 'FALSE')                   
                        worksheet.update_value('O'+str(source_index), tot_hours)
                        worksheet.update_value('Y'+str(source_index), 'DEV')
                else:
                    consecutive_alerts = 3
                    worksheet.update_value('W'+str(source_index), consecutive_alerts)
                    
            # Handling Outliers in the distribution - OTA (one time alerts)
            OTA = np.where((real < danger_lower) | (real > danger_upper), 'True', 'False')
            
                        
            #--------------------------------#            
            # Append info in the results table
            #--------------------------------#
            for nrows in range(num_hours):
                forecast = forecast.append(pd.DataFrame({'KPI_Code': [kpiCode], 'KPI': [kpiName], 'Date': [pd.to_datetime(str(start_date[nrows]))], 'Real_Values':[real[nrows]], 'Predictions':[pred[nrows]], 'Warning_Lower_Bound' : [warning_lower[nrows]],'Warning_Upper_Bound' : [warning_upper[nrows]],'Alert_Lower_Bound' : [danger_lower[nrows]],'Alert_Upper_Bound' : [danger_upper[nrows]], 'RMSE':[train_loss], 'RMSE_Test':[test_loss], 'Is_Warning':[is_warning[nrows]], 'Is_Alert':[is_danger], 'Rule_Based':[rule_based], 'OTA':[OTA[nrows]]}), sort=True)
                if nrows == 0:
                    realvalue_prediction = 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+'\n'
                else:
                    realvalue_prediction += 'Datetime: '+str(pd.to_datetime(str(start_date[nrows])).strftime("%m/%d/%Y %H:%M"))+' | Real Values: '+str(f'{round(real[nrows], 2):.2f}')+'\n'

            #----------------------------#
            # Alert Plot and Slack Message
            #----------------------------#
            if hours_to_plot > 0: 
                fig = Functions.plotAlertManagament(df1, hours_to_plot, kpiName, AM_upper_bound, AM_lower_bound, floor)
                if Local:
                    plt.savefig('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\Alert of '+str(kpiCode)+'of '+str(datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H"))+'.png')
                    plt.close()
                else:
                    plt_to_plot=tot_hours-hours_to_plot+1
                    Slack_Sender.Message_out_AlertManagament(kpiName, url[slack_channel], fig, plt_to_plot, tot_hours, realvalue_prediction)

        '''
        if Test:    
            if rule_based == 'True':
                fig = Functions.plotseries(df2, kpiName, problem_if, upper_bound, lower_bound)
            elif rule_based == 'False':
                fig = Functions.plotforecastTest(df1, kpiName, prediction, warning_ci, danger_ci,  confidence_interval, lower_bound_multi, upper_bound_multi, floor)
            elif rule_based == 'Alert_management':
                fig = Functions.plotAlertManagament(df1, hours_to_plot, kpiName, AM_upper_bound, AM_lower_bound, floor)
            fig
            fig.savefig('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\\'+str(is_danger)+' - Alert of '+str(kpiCode)+' in '+str(datetime.now().strftime("%d-%b-%Y_at_h%H"))+'.png')
        '''

    except:
        print('Error for '+str(kpiName))
        next
        
        
    #-------------------------------------------------------- PHASE 4 -------------------------------------------------#
    #------------------------------------------------------- SAVE DATA ------------------------------------------------#

    # resetting index for the upload
    forecast = forecast.reset_index(drop=True)


    if Local:
        forecast.to_csv('C:\\Users\\niccolo\\Desktop\\BI_Alert_System\\New_Release_v4\\Test_Results\\BI_Alert_System_'+str(datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H"))+'.csv', index=False)           
    else:
        ctx = DB_Connection.SN_connection()
        cursor = ctx.cursor()
        for index, row in forecast.iterrows(): # Update table MVMMART.PUBLIC.BI_ALERT_TEST with the results of the alerts
            if row['Rule_Based'] == 'True':
                try:
                    cursor.execute("INSERT INTO "+str(Database[Test])+" (KPI, TIMESTAMP, REAL_VALUES, PREDICTIONS, WARNING_LOWER_BOUND, WARNING_UPPER_BOUND, ALERT_LOWER_BOUND, ALERT_UPPER_BOUND, RMSE, RMSE_TEST, IS_WARNING, IS_ALERT, RUN_TIMESTAMP, IS_RULE_BASED, KPI_CODE, APP, SINGLE_ALERT) "  
                                   "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
                                   (str(row['KPI']),str(row['Date']),str(row['Real_Values']),None,None,None,str(row['Alert_Lower_Bound']),str(row['Alert_Upper_Bound']),None,None,str(row['Is_Warning']),str(row['Is_Alert']),str(datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H")+':00:00'),str(row['Rule_Based']),str(row['KPI_Code']),str(APP),str(row['OTA'])))
                    print('row '+str(index)+' ---- '+str(round((int(index)/(len(forecast)-1)*100),2))+'%')
                except:
                    print('error at row '+str(index)+' ---- '+str(round((int(index)/(len(forecast)-1)*100),2))+'%')
            else:
                try:
                    cursor.execute("INSERT INTO "+str(Database[Test])+" (KPI, TIMESTAMP, REAL_VALUES, PREDICTIONS, WARNING_LOWER_BOUND, WARNING_UPPER_BOUND, ALERT_LOWER_BOUND, ALERT_UPPER_BOUND, RMSE, RMSE_TEST, IS_WARNING, IS_ALERT, RUN_TIMESTAMP, IS_RULE_BASED, KPI_CODE, APP, SINGLE_ALERT) "  
                                   "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
                                   (str(row['KPI']),str(row['Date']),str(row['Real_Values']),str(row['Predictions']),str(row['Warning_Lower_Bound']),str(row['Warning_Upper_Bound']),str(row['Alert_Lower_Bound']),str(row['Alert_Upper_Bound']),str(row['RMSE']),str(row['RMSE_Test']),str(row['Is_Warning']),str(row['Is_Alert']),str(datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H")+':00:00'),str(row['Rule_Based']),str(row['KPI_Code']),str(APP),str(row['OTA'])))
                    print('row '+str(index)+' ---- '+str(round((int(index)/(len(forecast)-1)*100),2))+'%')
                except:
                    print('error at row '+str(index)+' ---- '+str(round((int(index)/(len(forecast)-1)*100),2))+'%')
        ctx.close()    
           