# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:37:45 2019

@author: niccolo
"""

import requests
import uuid
import os
import numpy as np

os.chdir("C:\\Users\\niccolo\\Desktop\\BI_Alert_System")
# Import Functions
import Functions_v4 as Functions



def Message_out_Predictions (kpiName, url, fig, tags, realvalue_prediction):
    
    filename = str(uuid.uuid4()) + ".png"
    fig.tight_layout()
    fig.savefig(filename)
    filepath = './'
    image_url = Functions.uploadFileAndGetUrl(filepath, filename)
    os.remove(str(filepath)+str(filename))
            
    #image_url = uploadFileAndGetUrl(filepath, filename)
    #url = 'https://hooks.slack.com/services/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' # Test
    headers = {'Content-type': 'application/json'}
    
    #tag_list = np.array(['<@wynn>','<@jamesh>','<@niccolo benvenuti>','<@kris>'])
    #to_tag = list(tag_list[tags])
    #for k in range(len(tag_list)):
    #    to_tag.append('')
    
    payload = {
            "attachments": [
                    {
                            "color": "danger",
                            "pretext": "ALERT: Abnormal Behaviour ", #+ to_tag[0] + ' ' + to_tag[1] + ' ' + to_tag[2] + ' '+ to_tag[3] + ' ' + to_tag[4] + ' ' + to_tag[5],
                            "author_name": "BI Team",
                            "title": "Link to Tableau",
                            "title_link": "https://10az.online.tableau.com/#/site/playstudios/views/MX20-BIAlertInfo/BIAlertBoundaries?:iid=1",
                            "text": str(kpiName)+" - Graph of the last 72 hours",
                            "image_url": image_url,
                            "fields": [
                                {
                                    "title": "Values - Fourier",
                                    "value": realvalue_prediction,
                                    "short": False
                                }                          
                            ]
                        }
                    ]
                }
    requests.post(url, headers=headers, json=payload).text
                
  
    
    
def Message_out_Rule_Based (kpiName, url, fig, tags, boundaries, realvalue_prediction):
    
    filename = str(uuid.uuid4()) + ".png"
    fig.tight_layout()
    fig.savefig(filename)
    filepath = './'
    image_url = Functions.uploadFileAndGetUrl(filepath, filename)
    os.remove(str(filepath)+str(filename))
           
    #image_url = uploadFileAndGetUrl(filepath, filename)
    #url = 'https://hooks.slack.com/services/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'
    headers = {'Content-type': 'application/json'}
    
    #tag_list = np.array(['<@wynn>','<@jamesh>','<@niccolo benvenuti>','<@kris>'])
    #to_tag = list(tag_list[tags])
    #for k in range(len(tag_list)):
    #    to_tag.append('')
    
    payload = {
            "attachments": [
                    {
                            "color": "danger",
                            "pretext": "ALERT: Abnormal Behaviour ", #+ to_tag[0] + ' ' + to_tag[1] + ' ' + to_tag[2] + ' '+ to_tag[3] + ' ' + to_tag[4] + ' ' + to_tag[5],
                            "author_name": "BI Team",
                            "title": "BI Alert",
                            "title_link": "https://10az.online.tableau.com/#/site/playstudios/views/MX20-BIAlertInfo/BIAlertBoundaries?:iid=1",
                            "text": str(kpiName)+" - Graph of the last 72 hours",
                            "image_url": image_url,
                            "fields": [
                                {
                                    "title": "Fixed Threshold",
                                    "value": boundaries,
                                    "short": True
                                },
                                {
                                    "title": "Values - Rule Based",
                                    "value": realvalue_prediction,
                                    "short": False
                                }        
                            ]
                        }
                    ]
                }
    requests.post(url, headers=headers, json=payload).text
      


          
  
def Message_out_AlertManagament (kpiName, url, fig, plt_to_plot, tot_hours, realvalue_prediction):
    
    filename = str(uuid.uuid4()) + ".png"
    fig.tight_layout()
    fig.savefig(filename)
    filepath = './'
    image_url = Functions.uploadFileAndGetUrl(filepath, filename)
    os.remove(str(filepath)+str(filename))
            
    #image_url = uploadFileAndGetUrl(filepath, filename)
    #url = 'https://hooks.slack.com/services/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' # Test
    headers = {'Content-type': 'application/json'}
    #tag_list = np.array(['<@wynn>','<@jamesh>','<@niccolo benvenuti>','<@kris>'])
    #to_tag = list(tag_list[tags])
    #for k in range(len(tag_list)):
    #    to_tag.append('')
    
    payload = {
            "attachments": [
                    {
                            "color": "#ffa500",
                            "pretext": "Alert Management", #+ to_tag[0] + ' ' + to_tag[1] + ' ' + to_tag[2] + ' '+ to_tag[3] + ' ' + to_tag[4] + ' ' + to_tag[5],
                            "author_name": "BI Team",
                            "title": "Link to Tableau",
                            "title_link": "https://10az.online.tableau.com/#/site/playstudios/views/MX20-BIAlertInfo/BIAlertBoundaries?:iid=1",
                            "text": str(kpiName)+" - "+str(plt_to_plot)+" of " +str(tot_hours)+ " planned snapshots",
                            "image_url": image_url,
                            "fields": [
                                {
                                    "title": "Values - Fourier",
                                    "value": realvalue_prediction,
                                    "short": False
                                }                          
                            ]
                        }
                    ]
                }
    requests.post(url, headers=headers, json=payload).text    
    
    
    

def Message_out_Back2Normal (kpiName, url, fig, tags, realvalue_prediction):
    
    filename = str(uuid.uuid4()) + ".png"
    fig.tight_layout()
    fig.savefig(filename)
    filepath = './'
    image_url = Functions.uploadFileAndGetUrl(filepath, filename)
    os.remove(str(filepath)+str(filename))
            
    #image_url = uploadFileAndGetUrl(filepath, filename)
    #url = 'https://hooks.slack.com/services/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' # Test
    headers = {'Content-type': 'application/json'}
    
    #tag_list = np.array(['<@wynn>','<@jamesh>','<@niccolo benvenuti>','<@kris>'])
    #to_tag = list(tag_list[tags])
    #for k in range(len(tag_list)):
    #    to_tag.append('')
    
    payload = {
            "attachments": [
                    {
                            "color": "#00ff00",
                            "pretext": "Good News: Everything's back to Normal ", #+ to_tag[0] + ' ' + to_tag[1] + ' ' + to_tag[2] + ' '+ to_tag[3] + ' ' + to_tag[4] + ' ' + to_tag[5],
                            "author_name": "BI Team",
                            "title": "Link to Tableau",
                            "title_link": "https://10az.online.tableau.com/#/site/playstudios/views/MX20-BIAlertInfo/BIAlertBoundaries?:iid=1",
                            "text": str(kpiName)+" - Graph of the last 72 hours",
                            "image_url": image_url,
                            "fields": [
                                {
                                    "title": "Values - Fourier",
                                    "value": realvalue_prediction,
                                    "short": False
                                }                          
                            ]
                        }
                    ]
                }
    requests.post(url, headers=headers, json=payload).text
    
    
    