import os

import shutil
import re
import math
import pandas as pd


 
def copyfile(srcfile,dstpath):                     
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)            
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                  
        shutil.copy(srcfile, os.path.join(dstpath , fname))          
        print ("copy %s -> %s"%(srcfile, os.path.join(dstpath , fname)))

def duplicatefile(srcname,dstname):                      
    if not os.path.isfile(srcname):
        print ("%s not exist!"%(srcname))
    else:
        fpath,fname=os.path.split(srcname)             
        shutil.copy(srcname, os.path.join(fpath , dstname))         
        print ("duplicate %s -> %s"%(srcname, os.path.join(fpath , dstname)))

def main():
    hasson_path =
    MNI_to_TRs_path = os.path.join(hasson_path,'Afni_Surface')
    for subj in os.listdir(MNI_to_TRs_path):
        dst=os.path.join(MNI_to_TRs_path,subj)
        global_norm_path = os.path.join(hasson_path,'Afni_Surface',subj)
        for task in os.listdir(global_norm_path):
            event_file_name='{}_task-{}_events.tsv'.format(subj,task)
            event_file=os.path.join(hasson_path,subj,'func',event_file_name)
            copyfile(event_file,dst)
            
def delete():
    hasson_path =
    MNI_to_TRs_path = os.path.join(hasson_path,'MNI_to_TRs')
    for subj in os.listdir(MNI_to_TRs_path):
        path=os.path.join(MNI_to_TRs_path,subj)
        for file_name in os.listdir(path):
            if os.path.isfile(os.path.join(path,file_name)):
                os.remove(os.path.join(path,file_name))
                print('delete:"{}'.format(os.path.join(path,file_name)))
                
                
def transcript_to_list(transcripts_path):
    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
    for transcirpt in os.listdir(transcripts_path):
        if '.txt' in transcirpt:
            file_path=os.path.join(transcripts_path,transcirpt)
            file = open(file_path,'r')  
            file_data = file.readlines() 
            data=[]
            for row in file_data:
                tmp_list = row.split(' ') 
                data=data+tmp_list 
            for word in data:
                word=re.sub(remove_chars,'',word)

def generate_gentle():
    root_path=
    stimuli_duration={}
    stimuli_path=
    for subject in os.listdir(root_path):
        all_task_path=os.path.join(root_path,subject)
        for task in os.listdir(all_task_path):
            task_path = os.path.join(all_task_path, task)
            if os.path.isdir(task_path):
                event_path=os.path.join(root_path,subject,'{}_task-{}_events.tsv'.format(subject,task))
                event=pd.read_csv(event_path,sep='\t')
                onset=event['onset'][0]
                stimuli=task
                if 'pieman' in task:
                    stimuli='pieman'
                if 'notthefall' in task:
                    stimuli='notthefallintact'
                if 'milkyway'==task:
                    stimuli='milkywayoriginal'
                if 'schema' in task:
                    continue
                gentle_path=os.path.join(stimuli_path,stimuli,'align.csv')
                gentle=pd.read_csv(gentle_path,header=None)
                for i in range(len(gentle)):
                    gentle[2][i]+=onset
                    gentle[3][i]+=onset
                save_gentle_path=os.path.join(root_path,subject,'{}_task-{}_align.csv'.format(subject,task))
                gentle.to_csv(save_gentle_path,header=False)
                        
            
            
            
if __name__ == '__main__':
	#main()
    generate_gentle()
