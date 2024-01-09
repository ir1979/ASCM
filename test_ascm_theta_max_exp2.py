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

    input_path=os.getcwd()
    folder='50k'
    files_path = glob.glob(input_path+'/data/'+folder+'/*.csv')  
    files_path.sort()
    # THETA=[1.01, 1.05, 1.1, 1.2, 1.5, 2, 5, 10]
    THETA=[20]
    MAX=[10]
    k=20
    columns=['dataset_name', 'method','k', 'theta','MAX','SSE','Total_time', 'allSSEs']
    all_results=pd.DataFrame()
    for file_path in files_path:
        
        results=pd.DataFrame(columns=columns)
        
        names= file_path.split('/')
        dataset_name=os.path.basename(names[-1])
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4] 
        
        data=pd.read_csv(file_path)
        a_2d=data.to_numpy()
        if len(data>50000):
            a_2d=a_2d[:50000]
        print(f'ASC method outputs for {dataset_name}....')
        for theta in THETA: 
            print(f'theta={theta}')
            for max in MAX:
                print(f'max={max}')
                start_a = time.time()
                sizes, SSE_a, all_SSEs, _ = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, max, theta, False)
                end_a = time.time()
                total_time_a= end_a-start_a
                
                asc_current= {'dataset_name':dataset_name, 'method':'asc', 'k':k, 
                              'theta':theta,'MAX':max, 'SSE':SSE_a, 'Total_time':total_time_a, 'allSSEs': all_SSEs}
                
                print(asc_current)
                results= results.append(asc_current, ignore_index=True)
                # results= pd.concat([results, asc_current], ignore_index=True)
                
                
                all_results= all_results.append(asc_current, ignore_index=True)
                # all_results= pd.concat([all_results, asc_current], ignore_index=True)

        
        results[columns].to_csv(input_path+f'/outs/{dataset_name}_exp2.csv')
        
        
    all_results[columns].to_csv(input_path+f'/outs/{folder}_exp2.csv')   
    print('Finished.')
if __name__ == "__main__":
    __main__(sys.argv)       


