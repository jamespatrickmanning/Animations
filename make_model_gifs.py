#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:08:51 2020

Combine pictures of model temperature ,then make gif

@author: jmanning & Mingchao
Modifed 3/3/2010 to clean up code
"""
import os
import animating_functions as af
from datetime import datetime,timedelta
import upload_modules as um

#Hardcodes############################
area = 'NorthShore'#get different gbox (see options in the "gbox" function below)
#model_name = 'GOMOFS' # styles such as DOPPIO and GOMOFS
model_name = 'DOPPIO'
start_date='2020-02-28'#local time
ndays=1  # keep in mind that DOPPIO has hourly fields while GOMOFS has 3-hourly fields
######################################
# form datetimes 
start_date_datetime=datetime(int(start_date[0:4]),int(start_date[5:7]),int(start_date[8:10]),0,0,0)
end_date_datetime=datetime(int(start_date[0:4]),int(start_date[5:7]),int(start_date[8:10]),0,0,0)+timedelta(days=ndays)
end_date=str(end_date_datetime.year)+'-'+str(end_date_datetime.month).zfill(2)+'-'+str(end_date_datetime.day).zfill(2)
# assign output directories relative to this one but this assumes all the code is in a directory called "py"
realpath=os.path.dirname(os.path.abspath(__file__))
if model_name == 'DOPPIO':
    dpath=realpath[::-1].replace('py'[::-1],'result/Doppio'[::-1],1)[::-1]  # the directory of the result but this assumes code is in a directoy called "py"
elif model_name == 'GOMOFS':
    dpath=realpath[::-1].replace('py'[::-1],'result/Gomofs'[::-1],1)[::-1]
if not os.path.exists(dpath):
    os.makedirs(dpath)
dictionary=os.path.join(dpath,'dictionary_emolt.p')
gif_path=os.path.join(dpath,'gif')
local_dir=os.path.join(dpath,'gif/')
map_save=os.path.join(dpath,'map')
gif_name = os.path.join(gif_path,start_date+area+'_'+model_name+'.gif')

#############################
 #run functions
af.seperate(filepathsave=dictionary)
#Get min/max temperature for color bar.Different models don't have same temperature range
Min_temp,Max_temp=af.temp_min_max(model_name,dt=start_date_datetime,interval=ndays,area=area)
#make images
af.make_images(model_name,dpath=dictionary,path=map_save,dt=start_date_datetime,interval=ndays,Min_temp=Min_temp,Max_temp=Max_temp,area=area)
#using images to make gif
af.make_gif(gif_name,map_save,start_time=start_date,end_time=end_date)
#um.sd2drf_update(local_dir,remote_dir='/anno_ftp/graphics')#the files will store in //66.114.154.52:8443/smb/file-manager/Home directory/anno_ftp/anno_ftp/graphics
