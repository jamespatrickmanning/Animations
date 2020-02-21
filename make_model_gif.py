#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:08:51 2020

@author: jmanning
"""
import os
import combine_models as cm
from datetime import datetime,timedelta

#Hardcodes
area = 'SNE'#get different gbox
model_name = 'gomofs' # styles such as doppio and gomofs
start_date='2020-02-11'
ndays=1
start_date_datetime=datetime(int(start_date[0:4]),int(start_date[5:7]),int(start_date[8:10]),0,0,0)
end_date_datetime=datetime(int(start_date[0:4]),int(start_date[5:7]),int(start_date[8:10]),0,0,0)+timedelta(days=ndays)
end_date=str(end_date_datetime.year)+'-'+str(end_date_datetime.month).zfill(2)+'-'+str(end_date_datetime.day).zfill(2)
realpath=os.path.dirname(os.path.abspath(__file__))
if model_name == 'doppio':
    dpath=realpath[::-1].replace('py'[::-1],'result/Doppio'[::-1],1)[::-1]  # the directory of the result
elif model_name == 'gomofs':
    dpath=realpath[::-1].replace('py'[::-1],'result/Gomofs'[::-1],1)[::-1]
if not os.path.exists(dpath):
    os.makedirs(dpath)
dictionary=os.path.join(dpath,'dictionary_emolt.p')
gif_path=os.path.join(dpath,'gif')
map_save=os.path.join(dpath,'map')
gif_name =os.path.join(gif_path,start_date+area+'_'+model_name+'.gif')

#############################
 #run functions
cm.seperate(filepathsave=dictionary)
cm.make_images(model_name,dpath=dictionary,path=map_save,dt=start_date_datetime,interval=ndays,area=area)
cm.make_gif(gif_name,map_save,start_time=start_date,end_time=end_date)
