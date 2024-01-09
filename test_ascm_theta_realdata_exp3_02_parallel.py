import sys
import time
import matplotlib.pyplot as plt

import accelerated_sequence_clustering as ascm
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import os
import concurrent.futures
from process_theta import process_theta 


# Get the number of available CPU cores
num_cores = os.cpu_count()
print(f'Number of CPU cores: {num_cores}')

def __main__(argv):

    input_path=os.getcwd()
    folder='data/real'
    files_path = glob.glob(input_path+'/'+folder+'/*.csv')  
    files_path.sort()
    theta_set=list(range(1,64))
    max_set = list(range(1,64))
    k_set=[20] 
    columns=['dataset_name', 'method','k', 'theta','max','SSE','Total_time', 'allSSEsBasic', 'allSSEsAccelerated', 'speed_up', 'Gap']
    all_results=pd.DataFrame()
    
    for file_path in files_path:
        
        results=pd.DataFrame(columns=columns)
        
        names= file_path.split('/')
        dataset_name=os.path.basename(names[-1])
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4] 
        
        data=pd.read_csv(file_path)
        a_2d=data.to_numpy()
        if len(data>5000):
            a_2d=a_2d[:5000]

        mx = np.max(a_2d,0)
        mn = np.min(a_2d,0)

        a_2d = (a_2d-mn)/(mx-mn)

        results = pd.DataFrame()

        print(f'Clustering outputs for {dataset_name}....')

        start_b = time.time()
        _, SSE_b, _, internal_left_matrix = ascm.basic_sequence_clustering_2d(a_2d, max(k_set), False)
        all_SSEsBasic = np.array(internal_left_matrix)[1:,-1]
        end_b = time.time()
        total_time_b= end_b-start_b

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:

            futures = [executor.submit(process_theta, dataset_name, a_2d, max(k_set), MAX, THETA, SSE_b, all_SSEsBasic, total_time_b) for MAX in max_set for THETA in theta_set]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

            # Process results if needed
            for future in futures:
                asc_current = future.result()
                results = pd.concat([results, pd.DataFrame([asc_current])], ignore_index=True)
       
        results[columns].to_csv(input_path+f'/outs3/{dataset_name}_exp3.csv')
        all_results = pd.concat([all_results, results], ignore_index=True)

    all_results[columns].to_csv(input_path+f'/outs3/exp3_2.csv')   
    print('Finished.')

if __name__ == "__main__":
    __main__(sys.argv)  