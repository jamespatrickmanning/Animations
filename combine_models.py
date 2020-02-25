#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:11:48 2020

@author: jmanning
"""
import os,imageio
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import time
import zlconversions as zl
import sys
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import math
from matplotlib import path
import copy

def get_doppio_url(dtime):
    '''dtime ids gmt time'''
    date=dtime.strftime('%Y-%m-%d')
    url='http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/runs/History_RUN_2018-11-12T00:00:00Z'
    return url.replace('2018-11-12',date)

def get_gomofs_url(date):
    """
    the format of date is:datetime.datetime(2019, 2, 27, 11, 56, 51, 666857)
    the date is the american time 
    input date and return the url of data
    """
#    print('start calculate the url!') 
    date=date+timedelta(hours=4.5)
    date_str=date.strftime('%Y%m%d%H%M%S')
    hours=int(date_str[8:10])+int(date_str[10:12])/60.+int(date_str[12:14])/3600.
    tn=int(math.floor((hours)/6.0)*6)  ## for examole: t12z the number is 12
    if len(str(tn))==1:
        tstr='t0'+str(tn)+'z'   # tstr in url represent hour string :t00z
    else:
        tstr='t'+str(tn)+'z'
    if round((hours)/3.0-1.5,0)==tn/3:
        nstr='n006'       # nstr in url represent nowcast string: n003 or n006
    else:
        nstr='n003'
    url='http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/GOMOFS/MODELS/'\
    +date_str[:6]+'/nos.gomofs.fields.'+nstr+'.'+date_str[:8]+'.'+tstr+'.nc'
    return url


def getgbox(area):
  # gets geographic box based on area
  if area=='SNE':
    gbox=[-70.,-64.,39.,42.] # for SNE
  elif area=='OOI':
    gbox=[-72.,-69.5,39.5,41.5] # for OOI
  elif area=='GBANK':
    gbox=[-70.,-64.,39.,42.] # for GBANK
  elif area=='GS':           
    gbox=[-71.,-63.,38.,42.5] # for Gulf Stream
  elif area=='NorthShore':
    gbox=[-71.,-69.5,41.5,43.] # for north shore
  elif area=='CCBAY':
    gbox=[-70.75,-69.8,41.5,42.23] # CCBAY
  elif area=='inside_CCBAY':
    gbox=[-70.75,-70.,41.7,42.23] # CCBAY
  elif area=='NEC':
    gbox=[-69.,-64.,39.,43.5] # NE Channel
  return gbox

def bbox2ij(lon,lat,bbox=[-160., -155., 18., 23.]):
    """Return indices for i,j that will completely cover the specified bounding box.     
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox)
    lon,lat = 2D arrays that are the target of the subset
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------  
    >>> i0,i1,j0,j1 = bbox2ij(lon_rho,[-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
    """
    bbox=np.array(bbox)
    mypath=np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(),lat.flatten())).T   
    n,m = np.shape(lon)
    inside = p.contains_points(points).reshape((n,m))
    #ii,jj = np.meshgrid(xrange(m),xrange(n))
    ii,jj = np.meshgrid(range(m),range(n))
    return min(ii[inside]),max(ii[inside]),min(jj[inside]),max(jj[inside])

def get_limited_gbox(area,lon,lat):
#def getgbox(area):
  # gets geographic box based on area
  if area=='SNE':
    gbox=[-70.,-64.,39.,42.] # for SNE
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='OOI':
    gbox=[-72.,-69.5,39.5,41.5] # for OOI
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='GBANK':
    gbox=[-70.,-64.,39.,42.] # for GBANK
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='GS':           
    gbox=[-71.,-63.,38.,42.5] # for Gulf Stream
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='NorthShore':
    gbox=[-71.,-69.5,41.5,43.] # for north shore
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='CCBAY':
    gbox=[-70.75,-69.8,41.5,42.23] # CCBAY
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='inside_CCBAY':
    gbox=[-70.75,-70.,41.7,42.23] # CCBAY
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  elif area=='NEC':
    gbox=[-69.,-64.,39.,43.5] # NE Channel
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox=gbox)
  #return gbox
  return i0,i1,j0,j1

def temp_min_max(model_name,dt=datetime(2019,5,1,0,0,0),interval=31,area='OOI'):
    ''' 
        Loop through each day to find min/max temp
        model_name:name of model
        dt: start time
        interval: how many days we need make
        area:limited area you want look
    '''
    temp_list=[]#store temperature of min and max
    if model_name == 'DOPPIO':
        interval=interval*24
        for j in range(interval):
            #dtime=dt+timedelta(days=j)
            dtime=dt+timedelta(hours=j)
            #print(dtime)
            url=get_doppio_url(dtime)
            while True:
                if zl.isConnected(address=url):
                    break
                print('check the website is well or internet is connected?')
                time.sleep(5)
            skip=0
            while True: 
                try:
                    nc = NetCDFFile(url)
                    lons=nc.variables['lon_rho'][:]
                    lats=nc.variables['lat_rho'][:]
                    temps=nc.variables['temp']
                    i0,i1,j0,j1 = get_limited_gbox(area,lon=lons,lat=lats)
                    break
                except RuntimeError:
                    print(str(url)+': need reread')
                except OSError:
                    if zl.isConnected(address=url):
                        print(str(url)+': file not exit.')
                        skip=1
                        break
                except KeyboardInterrupt:
                    sys.exit()
            if skip==1:
                continue
            #m_temp=mean_temp(temps)# here we are taking a daily average
            m_temp=temps[np.mod(j,24),0]#0 is bottom of depth,-1 is surface of depth
            ntime=dtime
            #time_str=ntime.strftime('%Y-%m-%d')
            temp=m_temp*1.8+32
            temp_F = temp[j0:j1, i0:i1]
            Min_temp=int(min(temp_F.data[~np.isnan(temp_F.data)]))
            Max_temp=int(max(temp_F.data[~np.isnan(temp_F.data)]))
            temp_list.append(Min_temp)
            temp_list.append(Max_temp)
    elif model_name == 'GOMOFS':
        for j in range(interval): # loop every days files 
            dtime=dt+timedelta(days=j)
            #print(dtime)
            skip=0 #count use to count how many files load successfully
            for i in range(0,24,3): #loop every file of day, every day have 8 files
                ntime=dtime+timedelta(hours=i)
                url=get_gomofs_url(ntime)
                print(url)
                while True:#check the internet
                    if zl.isConnected(address=url):
                        break
                    print('check the website is well or internet is connected?')
                    time.sleep(5)
                while True:  #load data
                    try:
                        nc = NetCDFFile(url)
                        lons=nc.variables['lon_rho'][:]
                        lats=nc.variables['lat_rho'][:]
                        temps=nc.variables['temp']
                        i0,i1,j0,j1 = get_limited_gbox(area,lon=lons,lat=lats)
                        break
                    except KeyboardInterrupt:
                        sys.exit()
                    except OSError:
                        if zl.isConnected(address=url):
                            print(str(url)+': file not exit.')
                            skip=1
                            break
                    except:
                        print('reread data:'+str(url))
                if skip==1:  #if file is not exist   
                    continue
                m_temp=temps[0,0] # JiM added this 2/19/2020
                #m_temp=m_temp/float(count)
                #ntime=dtime
                temp=m_temp*1.8+32
                temp_F = temp[j0:j1, i0:i1]
                Min_temp=int(min(temp_F.data[~np.isnan(temp_F.data)]))
                #Mingchao created a deepcopy for filtering the wrong max temperature ,such as 1e+37(9999999999999999538762658202121142272)
                b=copy.deepcopy(list(temp_F.data[~np.isnan(temp_F.data)]))
                for k in range(len(np.where(temp_F.data[~np.isnan(temp_F.data)]>100)[0])):
                    b.remove(int(list(temp_F.data[~np.isnan(temp_F.data)])[np.where(temp_F.data[~np.isnan(temp_F.data)]>100)[0][k]]))
                Max_temp=int(max(list(b)))
                temp_list.append(Min_temp)
                temp_list.append(Max_temp)
    Min_temp = min(temp_list)
    Max_temp = max(temp_list)
    return Min_temp,Max_temp

def plotit(model_name,lons,lats,slons,slats,temp,depth,time_str,path_save,dpi=80,Min_temp=0,Max_temp=0,area='OOI'):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_axes([0.01,0.05,0.98,0.87])
    # create polar stereographic Basemap instance.
    gb=getgbox(area)
    m = Basemap(projection='stere',lon_0=(gb[0]+gb[1])/2.,lat_0=(gb[2]+gb[3])/2.,lat_ts=0,llcrnrlat=gb[2],urcrnrlat=gb[3],\
                llcrnrlon=gb[0],urcrnrlon=gb[1],rsphere=6371200.,resolution='f',area_thresh=100)# JiM changed resolution to "c" for crude
    #m = Basemap(projection='stere',lon_0=-70.25,lat_0=40.5,lat_ts=0,llcrnrlat=37,urcrnrlat=44,\
    #            llcrnrlon=-75.5,urcrnrlon=-65,rsphere=6371200.,resolution='i',area_thresh=100)# JiM changed resolution to "c" for crude
                #llcrnrlon=-75.5,urcrnrlon=-65,rsphere=6371200.,resolution='c',area_thresh=100)
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    if len(slons)!=0:
        x1,y1=m(slons,slats)
        ax.plot(x1,y1,'ro',markersize=10)
    # draw parallels.
    parallels = np.arange(0.,90,1.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=20)
    # draw meridians
    meridians = np.arange(180.,360.,1.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=20)
    x, y = m(lons, lats) # compute map proj coordinates.
    # draw filled contours.
    
    
    dept_clevs=[50,100, 150,300,1000]
    dept_cs=m.contour(x,y,depth,dept_clevs,colors='black')
    plt.clabel(dept_cs, inline = True, fontsize =15,fmt="%1.0f")
    
    
    #clevs=np.arange(35.,59.,0.5)  #for all year:np.arange(34,84,1) or np.arange(34,68,1)
    clevs=np.arange(Min_temp,Max_temp,0.5)
    cs = m.contourf(x,y,temp,clevs,cmap=plt.get_cmap('rainbow'))
    # add colorbar.
    cbar = m.colorbar(cs,location='right',pad="2%",size="5%")
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
    cbar.set_label('Fahrenheit',fontsize=25)
    # add title
    plt.title(model_name+' MODEL BOTTOM TEMPERATURE '+time_str,fontsize=24)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    plt.savefig(os.path.join(path_save,time_str.replace(' ','t')+'.png'),dpi=dpi)

def mean_temp(temps):
    mean_temp=temps[0,0]
    for i in range(1,24):
        mean_temp+=temps[i,0]# note we are using the bottom level 0
    return mean_temp/24.0


def make_images(model_name,dpath,path,dt=datetime(2019,5,1,0,0,0),interval=31,Min_temp=0,Max_temp=10,area='OOI'):
    '''dpath: the path of dictionary, use to store telemetered data
        path: use to store images
        dt: start time
        interval: how many days we need make 
    '''
    with open(dpath,'rb') as fp:
         telemetered_dict=pickle.load(fp)
    if model_name == 'DOPPIO':
        interval=interval*24
        for j in range(interval):
            #dtime=dt+timedelta(days=j)
            dtime=dt+timedelta(hours=j)
            print(dtime)
            url=get_doppio_url(dtime)
            while True:
                if zl.isConnected(address=url):
                    break
                print('check the website is well or internet is connected?')
                time.sleep(5)
            skip=0
            while True: 
                try:
                    nc = NetCDFFile(url)
                    lons=nc.variables['lon_rho'][:]
                    lats=nc.variables['lat_rho'][:]
                    temps=nc.variables['temp']
                    depth=nc.variables['h'][:]
                    #i0,i1,j0,j1 = get_limited_gbox(area,lon=lons,lat=lats)
                    break
                except RuntimeError:
                    print(str(url)+': need reread')
                except OSError:
                    if zl.isConnected(address=url):
                        print(str(url)+': file not exit.')
                        skip=1
                        break
                except KeyboardInterrupt:
                    sys.exit()
            if skip==1:
                continue
            #m_temp=mean_temp(temps)# here we are taking a daily average
            m_temp=temps[np.mod(j,24),0]#0 is bottom of depth,-1 is surface of depth
            ntime=dtime
            #time_str=ntime.strftime('%Y-%m-%d')
            time_str=ntime.strftime('%Y-%m-%d-%H')
            temp=m_temp*1.8+32
            #temp_F = temp[j0:j1, i0:i1]
            #Min_temp=int(min(temp_F.data[~np.isnan(temp_F.data)]))
            #Max_temp=int(max(temp_F.data[~np.isnan(temp_F.data)]))
            Year=str(ntime.year)
            Month=str(ntime.month)
            Day=str(ntime.day)
            slons,slats=[],[]
            try:
                slons,slats=[],[]
                for i in telemetered_dict[Year][Month][Day].index:
                    slons.append(telemetered_dict[Year][Month][Day]['lon'].iloc[i])
                    slats.append(telemetered_dict[Year][Month][Day]['lat'].iloc[i])
            except:
                slons,slats=[],[]
            dpi=80
            plotit(model_name,lons,lats,slons,slats,temp,depth,time_str,path,dpi,Min_temp,Max_temp,area)
    if model_name == 'GOMOFS':
        for j in range(interval): # loop every days files 
            dtime=dt+timedelta(days=j)
            print(dtime)
            count,skip=0,0  #count use to count how many files load successfully
            #skip=0
            for i in range(0,24,3): #loop every file of day, every day have 8 files
                ntime=dtime+timedelta(hours=i)
                url=get_gomofs_url(ntime)
                print(url)
                while True:#check the internet
                    if zl.isConnected(address=url):
                        break
                    print('check the website is well or internet is connected?')
                    time.sleep(5)
                while True:  #load data
                    try:
                        nc = NetCDFFile(url)
                        lons=nc.variables['lon_rho'][:]
                        lats=nc.variables['lat_rho'][:]
                        temps=nc.variables['temp']
                        depth=nc.variables['h'][:]
                        #i0,i1,j0,j1 = get_limited_gbox(area,lon=lons,lat=lats)
                        break
                    except KeyboardInterrupt:
                        sys.exit()
                    except OSError:
                        if zl.isConnected(address=url):
                            print(str(url)+': file not exit.')
                            skip=1
                            break
                    except:
                        print('reread data:'+str(url))
                if skip==1:  #if file is not exist   
                    continue
                m_temp=temps[0,0] # JiM added this 2/19/2020
                '''
                if i==0: 
                    count+=1
                    m_temp=temps[0,0]
                else:
                    m_temp+=temps[0,0]
                    count+=1
                '''
                #m_temp=m_temp/float(count)
                #ntime=dtime
                time_str=ntime.strftime('%Y-%m-%d-%H')
                temp=m_temp*1.8+32
                #temp_F = temp[j0:j1, i0:i1]
                #Min_temp=int(min(temp_F.data[~np.isnan(temp_F.data)]))
                #Mingchao created a deepcopy for filtering the wrong max temperature ,such as 1e+37(9999999999999999538762658202121142272)
                #b=copy.deepcopy(list(temp_F.data[~np.isnan(temp_F.data)]))
                #for k in range(len(np.where(temp_F.data[~np.isnan(temp_F.data)]>100)[0])):
                    #b.remove(int(list(temp_F.data[~np.isnan(temp_F.data)])[np.where(temp_F.data[~np.isnan(temp_F.data)]>100)[0][k]]))
                #Max_temp=int(max(list(b)))
                Year=str(ntime.year)
                Month=str(ntime.month)
                Day=str(ntime.day)
                slons,slats=[],[]
                try:
                    slons,slats=[],[]
                    for i in telemetered_dict[Year][Month][Day].index:
                        slons.append(telemetered_dict[Year][Month][Day]['lon'].iloc[i])
                        slats.append(telemetered_dict[Year][Month][Day]['lat'].iloc[i])
                except:
                    slons,slats=[],[]
                dpi=80
                plotit(model_name,lons,lats,slons,slats,temp,depth,time_str,path,dpi,Min_temp,Max_temp,area)
        
def read_telemetry(path):
    """read the telemetered data and fix a standard format, the return the standard data"""
    tele_df=pd.read_csv(path,sep='\s+',names=['vessel_n','esn','month','day','Hours','minates','fracyrday',\
                                          'lon','lat','dum1','dum2','depth','rangedepth','timerange','temp','stdtemp','year'])
    if len(tele_df)<6000:
        print('Warning: the emolt.dat file is not complete at this time.')
        #sys.exit()
        
    return tele_df


def seperate(filepathsave,ptelemetered='https://www.nefsc.noaa.gov/drifter/emolt.dat'):
    '''create a dictionary use to store the data from telemetered, index series is year, month, day and hour
    ptelemetered: the path of telemetered
    '''
    dfdict={}
    df=read_telemetry(ptelemetered)
    for i in df.index:
        if df['depth'][i]<2.0:
            continue
        if df['minates'].iloc[i]<=30:
            Ctime=datetime.strptime(str(df['year'].iloc[i])+'-'+str(df['month'].iloc[i])+'-'+str(df['day'].iloc[i])+' '+\
                                         str(df['Hours'].iloc[i])+':'+str(df['minates'].iloc[i])+':'+'00','%Y-%m-%d %H:%M:%S')
        else:
            Ctime=datetime.strptime(str(df['year'].iloc[i])+'-'+str(df['month'].iloc[i])+'-'+str(df['day'].iloc[i])+' '+\
                                         str(df['Hours'].iloc[i])+':'+str(df['minates'].iloc[i])+':'+'00','%Y-%m-%d %H:%M:%S')+timedelta(seconds=1800)
        Year=str(Ctime.year)
        Month=str(Ctime.month)
        Day=str(Ctime.day)
        if not Year in dfdict:
            dfdict[Year]={}
        if not Month in dfdict[Year]:
            dfdict[Year][Month]={}
        if not Day in dfdict[Year][Month]:
            dfdict[Year][Month][Day]={}

        if len(dfdict[Year][Month][Day])!=0:
            dfdict[Year][Month][Day]=dfdict[Year][Month][Day].append(pd.DataFrame(data=[[df['lat'].iloc[i],df['lon'].iloc[i],df['temp'].iloc[i]]],columns=['lat','lon','temp']).iloc[0])
            dfdict[Year][Month][Day].index=range(len(dfdict[Year][Month][Day]))
        else:
            dfdict[Year][Month][Day]=pd.DataFrame(data=[[df['lat'].iloc[i],df['lon'].iloc[i],df['temp'].iloc[i]]],columns=['lat','lon','temp'])
    with open(filepathsave,'wb') as fp:
        pickle.dump(dfdict,fp,protocol=pickle.HIGHEST_PROTOCOL)


def make_gif(gif_name,png_dir,start_time=False,end_time=False,frame_length = 0.2,end_pause = 4 ):
    '''use images to make the gif
    frame_length: seconds between frames
    end_pause: seconds to stay on last frame
    the format of start_time and end time is string, for example: %Y-%m-%d(YYYY-MM-DD)'''
    
    if not os.path.exists(os.path.dirname(gif_name)):
        os.makedirs(os.path.dirname(gif_name))
    allfile_list = glob.glob(os.path.join(png_dir,'*.png')) # Get all the pngs in the current directory
    print(allfile_list)
    file_list=[]
    if start_time:    
        for file in allfile_list:
            if start_time<=os.path.basename(file).split('.')[0]<=end_time:
                file_list.append(file)
    else:
        file_list=allfile_list
    list.sort(file_list, key=lambda x: x.split('/')[-1].split('t')[0]) # Sort the images by time, this may need to be tweaked for your use case
    images=[]
    # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
    for ii in range(0,len(file_list)):       
        file_path = os.path.join(png_dir, file_list[ii])
        if ii==len(file_list)-1:
            for jj in range(0,int(end_pause/frame_length)):
                images.append(imageio.imread(file_path))
        else:
            images.append(imageio.imread(file_path))
    # the duration is the time spent on each image (1/duration is frame rate)
    imageio.mimsave(gif_name, images,'GIF',duration=frame_length)
