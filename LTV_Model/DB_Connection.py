# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:36:12 2020

@author: niccolo
"""

import snowflake.connector
import pandas as pd

# Connection String to Snowflake Database
def SN_connection():
    
    ctx = snowflake.connector.connect(
            user='',
            password='',
            account='',
            role='',
            database='',
            warehouse=''
            )
    return ctx