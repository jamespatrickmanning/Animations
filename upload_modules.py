#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:51:21 2019

@author: leizhao
"""
import ftplib 
import os
import zlconversions as zl



def sd2drf_update(local_dir,remote_dir):
    '''function: upload the local file to the student drifters
    input:
        local_dir: local directory
        remote_dir: remote directory,the folder in the student drifters'''
    #determine whether the string of the path is right 
    if local_dir[0]!='/':
        local_dir='/'+local_dir
    if remote_dir[0]!='/':
        remote_dir='/'+remote_dir
    upflist=zl.list_all_files(local_dir)  # get all file paths and name in local directory
    ftp=ftplib.FTP('66.114.154.52','huanxin','123321')   #logge in student drifters
    print('the number of upload files:'+str(len(upflist)))
    ftp.quit()
    if len(upflist)==0: # If there is no file to upload, then return a value of 0
        return 0
    for file in upflist:  #loop every file, upload file
        ftp=ftplib.FTP('66.114.154.52','huanxin','123321')
        fpath,fname=os.path.split(file)
        remote_dir_file=file.replace(fpath,remote_dir)
        dir=remote_dir   
        mkds(dir,ftp)  #if there is no folder in student drifters, create a new one.
        ftp_upload(file,remote_dir_file,ftp)  # upload file
        ftp.quit() 

def sd2drf(local_dir,remote_dir,filetype='**',keepfolder=False):
    '''function: Upload all files under one folder (including all files under subfolders) to the specified folder 
    input:
        local_dir: local directory
        remote_dir: remote directory,the folder in the student drifters
        keepfolder: wheather need keep subdirectory, if we need let the value is True'''
    
    if local_dir[0]!='/':
        local_dir='/'+local_dir
    if remote_dir[0]!='/':
        remote_dir='/'+remote_dir
    cdflist=zl.list_all_files(local_dir)
    files=[]
    if filetype=='**':#upload all files
        files=cdflist
    else:    #filter out the specified format file
        for file in cdflist:
            if file.split('.')[1] in filetype:
                files.append(file)
    ftp=ftplib.FTP('66.114.154.52','huanxin','123321')
    drifterlist=list_ftp_allfiles(remote_dir,ftp) #get all filename and file path in the student drifters 
    drflist=[]
    if keepfolder:#keep subdirectory
        for i in range(len(drifterlist)):  #change the path, use to detemine which files is not exist in student drifters 
            drflist.append(drifterlist[i].replace(remote_dir,local_dir))
        upflist=list(set(files)-set(drflist))  #caculate the files that is not exist in student drifters
        print(len(upflist))
        ftp.quit()
        if len(upflist)==0: #return 0 if there is no file need upload. 
            return 0
        for file in upflist:  #start loop files that need to upload
            ftp=ftplib.FTP('66.114.154.52','huanxin','123321')
            fpath,fname=os.path.split(file)  #seperate the filename and file path
            remote_dir_file=file.replace(local_dir,remote_dir) #initial the file path and name in student drifters
            dir=fpath.replace(local_dir,remote_dir).replace('//','/') #get the path of student drifters that this file should exist
            mkds(dir,ftp)  #check the path whether exist, if not make a new one
            ftp_upload(file,remote_dir_file,ftp)  #upload file
            ftp.quit() 
    else:  #just upload files,cancel subfolder
        for file in drifterlist:  #loop every filepath and filename, collect every filename 
            fpath,fname=os.path.split(file) #seperate the filename and file path
            drflist.append(fname)# add the filename to the drflist(the list include filenames that under student drifters' directory)
        #filter out which files need upload
        upflist=[]
        for file in files:
            fpath,fname=os.path.split(file)
            if fname not in drflist:  # if this file is not exist in student drifter, that file need add to uplist(upload list)
                upflist.append(file)
        
        print('the number of upload files:'+str(len(upflist)))
        ftp.quit()
        if len(upflist)==0:  #return 0 if there is no file need upload. 
            return 0
        for file in upflist:
            ftp=ftplib.FTP('66.114.154.52','huanxin','123321')
            fpath,fname=os.path.split(file)
            remote_dir_file=file.replace(fpath,remote_dir)
            dir=remote_dir   
            mkds(dir,ftp) #check the path whether exist, if not make a new one
            ftp_upload(file,remote_dir_file,ftp)
            ftp.quit()  


def directory_exists(dir,ftp):
    '''function:judge the directory is exist or not
    input:
        dir: the directory in student drifters
        ftp: for examle:ftplib.FTP('66.114.154.52','huanxin','123321')
    '''
    filelist = []
    ftp.retrlines('LIST',filelist.append)  # get the list of the files and foders
    for f in filelist:
        if f.split()[-1] == dir and f.upper().startswith('D'):#determine whether the directory is exist
            return True
    return False

def chdir(dir,ftp): 
    '''function:Change directories - create it if it doesn't exist
    input:
        dir: the directory
        '''
    if directory_exists(dir,ftp) is False: #IF the directory is not exist, we need create a new one
        ftp.mkd(dir) #create the new directory
        print(dir)
    ftp.cwd(dir)  #enter the folder
def ftp_upload(localfile, remotefile,ftp):
    '''funtion: upload local file to sudent drifters
    input: 
        localfile: the string that include local file path and name 
        remotefile: the string that include the path and filename that we set in student drifeters
    '''
    fp = open(localfile, 'rb')  # open local file
    ftp.storbinary('STOR %s' % os.path.basename(localfile), fp, 1024)  # upload file
    fp.close()
    print ("after upload " + localfile + " to " + remotefile)
  
def mkds(dir,ftp):
    '''function: create the new folder in student drifter
    input:
        dir: the string that we neeed create folder path'''
    dir_list=dir.split('/')
    for i in range(len(dir_list)):  #determine whether the folder that we need create
        if len(dir_list[i])==0:
            continue
        else:
            chdir(dir_list[i],ftp)        
def list_ftp_allfiles(rootdir,ftp):
    """get all files' path and name in rootdirectory
    rootdir: the root directory"""
    ftp.cwd('/')  #back to main directory
    ftp.cwd(rootdir)  #entor the root directory
    list = ftp.nlst()  #get the list of folders name and files name under the root directory
    _files = []
    for i in range(len(list)):#get the string of path and file name
        try:
            path=os.path.join(rootdir,list[i])
            _files.extend(list_ftp_allfiles(path,ftp))
        except ftplib.error_perm:
            path=os.path.join(rootdir,list[i])
            _files.append(path)
    return _files
