
import time
import matplotlib.pyplot as plt

import accelerated_sequence_clustering as ascm
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import os
import sys
def __main__(argv):

    # input_path=f'C:/Users/Elham/Dropbox/PhD Thesis/Anomaly Detection in Time series/projects/DataSets/Physionet/'
    # folder='abdominal-and-direct-fetal-ecg-database-1.0.0'
    # folder='brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0'
    # input_path='C:/Users/Elham/Dropbox/PhD Thesis/Anomaly Detection in Time series/projects/12 - Accelerated Sequential Clustering Module (ASCM)/'
    input_path=os.getcwd()
    folders=['5k','10k', '50k','100k_1','100k_2']
    folder=folders[0]
    files_path = glob.glob(input_path+'/data/'+folder+'/*.csv')  
    files_path.sort()
    theta=20
    MAX=10
    # k_min = 3
    # k_max = 20
    k_set=[3,5,10,20]
    columns=['dataset_name', 'method', 'k','SSE_a','Total_time_a']
    all_results=pd.DataFrame()
    
    # files_path=[input_path+folder+'/01_.csv']
    print(f'exp1 for {folder} time seris....')
    for file_path in files_path:
        
        results=pd.DataFrame(columns=columns)
        
        names= file_path.split('/')
        dataset_name=os.path.basename(names[-1])
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4] 
        
        data=pd.read_csv(file_path)
        a_2d=data.to_numpy()
        # a_2d=a_2d[:50000]
        print(f'Basic method outputs for {dataset_name}....')
        # for k in k_set: # range(k_min, k_max+1): # 
        #     print(f'cluster={k}')
            # start_b = time.time()
            # sizes, SSE_b, _, _ = ascm.basic_sequence_clustering_2d(a_2d, k, True)
            # end_b = time.time()
            # total_time_b= end_b-start_b
            # basic_current= {'dataset_name':dataset_name, 'method':'basic', 'k':k,'SSE':SSE_b,
            #                 'Total_time':total_time_b}
            # results= results.append(basic_current, ignore_index=True)
            # all_results= all_results.append(basic_current, ignore_index=True)
        
        print(f'ASC method outputs for {dataset_name}....')
        for k in k_set: #range(k_min, k_max+1):
            print(f'cluster={k}')
            start_a = time.time()
            sizes, SSE_a, _, _ = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, MAX, theta,False)
            end_a = time.time()
            total_time_a= end_a-start_a
            
            asc_current= {'dataset_name':dataset_name, 'method':'asc', 'k':k,'SSE_a':SSE_a,
                        'Total_time_a':total_time_a}
            results= results.append(asc_current, ignore_index=True)
            all_results= all_results.append(asc_current, ignore_index=True)
            # speed_up=total_time_b/total_time_a
            # Gap=(SSE_a-SSE_b)/SSE_b*100
            # print(f'speed up={speed_up}')
            # print(f'Gap={Gap}')
            # print('end')
      
        results[columns].to_csv(input_path+'/outs2/exp1'+f'/{dataset_name}.csv')
        
        
    all_results[columns].to_csv(input_path+'/outs2/exp1'+f'/{folder}.csv')   
    print(f'Finished for {folder} time seris.')
if __name__ == "__main__":
    __main__(sys.argv)       


