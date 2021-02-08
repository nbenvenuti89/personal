# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:30:15 2020

@author: niccolo
"""


# Import Libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
#import datetime as dt
import pygsheets

# Get Current Directory
os.chdir("C:\\Users\\niccolo\\Desktop\\BI_Alert_System")
#os.getcwd()
# Import Functions
import DB_Connection 



Last14Days = True
upperbound = 0

###########################################
# Downloading Tables
###########################################

# import Kpi Table
sql_globalKpis = []
globalKpis = []

gc = pygsheets.authorize(service_file='.//New_Release_v4//Python-AUTH-dfd0f213ea4e.json')
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
        sql_globalKpis = ', '.join(globalKpis) # Generate the SQL that gets all KPI data at just one query. group by platform, location, appversion
raw_kpi_table2 = raw_kpi_table[raw_kpi_table["if_invalid"]==''].reset_index().replace('NA', np.NaN)


# Aurora Connection
ctx_Aurora = DB_Connection.AURORA_connection()
# Return the past 10 days' data, group by platform, location, appversion
df_global = pd.read_sql('''
select CONVERT_TZ(TIMESTAMP(date,concat(HR,':00:00')),'America/Los_Angeles','Asia/Hong_Kong') as LOCAL_HOUR, ''' + sql_globalKpis + ''' from nrtreporting.mx_magnum_bi 
group by 1 order by 1
'''  , ctx_Aurora) # -- where version like '3.%'  where date >= date_sub(CURRENT_DATE(), INTERVAL 11 DAY) 
ctx_Aurora.close()
df_global.set_index('LOCAL_HOUR', inplace=True)

ctx_Asia = DB_Connection.SN_connection()
df_history = pd.read_sql('''
select TIMESTAMP, KPI_CODE from mvmmart.public.bi_alert_test where rank = 1 and is_alert = 'True' and timestamp>='2020-02-03'
'''  , ctx_Asia) # where date >= date_sub(CURRENT_DATE(), INTERVAL 11 DAY) 
ctx_Asia.close()
    
    

if Last14Days:
    for i in (df_global):
        dglob = df_global.iloc[len(df_global)-336-upperbound:len(df_global)-upperbound,:][i]
        dhist = df_history[df_history.KPI_CODE == i]['TIMESTAMP']
        kpi = raw_kpi_table2[raw_kpi_table2.kpi_code==i]
        if len(dhist)>0:
            fig = plt.figure(figsize=(20.5,13.5))
            plt.plot(dglob.index, dglob)
            for j in dhist:
                plt.axvline(x=j, color='r',ls='-')
            plt.xticks(rotation = '30')
            plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            plt.title(str(i)+' - is rule based = '+str(kpi.rule_based.to_string(index=False))+' - is alert if '+str(kpi.problem_if.to_string(index=False)), fontsize = 20)
            plt.savefig('./New_Release_v4/Test_Results/Testing_Alert/KPI of '+str(i)+'.png')
            plt.close()
        else:
            fig = plt.figure(figsize=(20.5,13.5))
            plt.plot(dglob.index, dglob)
            plt.xticks(rotation = '30')
            plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            plt.title(str(i)+' - is rule based = '+str(kpi.rule_based.to_string(index=False))+' - is alert if '+str(kpi.problem_if.to_string(index=False)), fontsize = 20)
            plt.savefig('./New_Release_v4/Test_Results/Testing_Alert/NO_Alert_KPI of '+str(i)+'.png')
            plt.close()
       
            
            
            
'''            
df_global = df_global.iloc[104:len(df_global),:].reset_index(drop=True)           
fig = plt.figure(figsize=(20.5,13.5))
plt.plot(np.arange(0,len(df_global['Lp_lock_percentage']),1), df_global['Lp_lock_percentage'])
plt.xticks(np.arange(0,len(df_global['Lp_lock_percentage']), step=12), df_global['LOCAL_HOUR'][np.arange(0,len(df_global['Lp_lock_percentage']),step=12)].dt.strftime("%m/%d %H:%M"), rotation = '30')
plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
plt.title(str('Lp_lock_percentage'), fontsize = 20)
plt.show()
'''