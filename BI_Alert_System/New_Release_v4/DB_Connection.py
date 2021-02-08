# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:48:45 2019

@author: niccolo
"""

import snowflake.connector
import base64
import pymysql


# Connection String to Snowflake Database
def SN_connection():
    
    pwd = base64.b64decode(b'amFiYmFy==') 
    ctx = snowflake.connector.connect(
            user='BI_PROJECTS',
            password=pwd.decode('utf-8'),
            account='playstudiosasia',
            role='BI_PROJECTS',
            database='MVMMART',
            warehouse='ANALYSTWH'
            )
    return ctx



# Connection String to Aurora (Amazon Redshift) Database
def AURORA_connection():

    host='psbicluster.cluster-cflptvu0j15e.us-west-2.rds.amazonaws.com'
    port=3306
    user=base64.b64decode(b'bmljY29sb2I=')
    password=base64.b64decode(b'bmljY29sb2I=')
    ctx = pymysql.connect(host,user=user,port=port,passwd=password)
    return ctx

