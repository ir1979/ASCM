
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
    folder='datasets3'
    files_path = glob.glob(input_path+folder+'/*.csv')  
    files_path.sort()
    theta=1.05
    MAX=10
    k_min = 3
    k_max = 20
    k_set=[3,5,10,20]
    columns=['dataset_name', 'method', 'k','SSE','Total_time']
    all_results=pd.DataFrame()
    
    # files_path=[input_path+folder+'/01_.csv']

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
        for k in  k_set: # range(k_min, k_max+1):
            print(f'cluster={k}')
            start_b = time.time()
            sizes, SSE_b, _, _ = ascm.basic_sequence_clustering_2d(a_2d, k)
            end_b = time.time()
            total_time_b= end_b-start_b
            basic_current= {'dataset_name':dataset_name, 'method':'basic', 'k':k,'SSE':SSE_b,
                            'Total_time':total_time_b}
            results= results.append(basic_current, ignore_index=True)
            all_results= all_results.append(basic_current, ignore_index=True)
        
        print(f'ASC method outputs for {dataset_name}....')
        for k in k_set: #range(k_min, k_max+1):
            print(f'cluster={k}')
            start_a = time.time()
            sizes, SSE_a = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, MAX, theta)
            end_a = time.time()
            total_time_a= end_a-start_a
            
            asc_current= {'dataset_name':dataset_name, 'method':'asc', 'k':k,'SSE':SSE_a,
                        'Total_time':total_time_a}
            results= results.append(asc_current, ignore_index=True)
            all_results= all_results.append(asc_current, ignore_index=True)

        output_path=os.getcwd()
        results[columns].to_csv(output_path+f'/{dataset_name}.csv')
        
        
    all_results[columns].to_csv(output_path+f'/{folder}.csv')   
    print('Finished.')
if __name__ == "__main__":
    __main__(sys.argv)       


