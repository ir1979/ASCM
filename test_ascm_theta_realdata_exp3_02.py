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
    folder='data/real'
    files_path = glob.glob(input_path+'/'+folder+'/*.csv')  
    files_path.sort()
    THETA=[1,2,4,8,16,32,64]
    max=10
    k_set=[20] # ]ist(range(3,21))
    columns=['dataset_name', 'method','k', 'theta','max','SSE','Total_time', 'allSSEs']
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

        mx = np.max(a_2d,0)
        mn = np.min(a_2d,0)

        a_2d = (a_2d-mn)/(mx-mn)

        print(f'ASC method outputs for {dataset_name}....')
        for k in k_set: 
            print(f'k={k}')
            for theta in THETA:
                print('Theta: ' + str(theta))
                start_a = time.time()
                sizes, SSE_a, all_SSEs, _ = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, max, theta, False)
                end_a = time.time()
                total_time_a= end_a-start_a
                
                asc_current= {'dataset_name':dataset_name, 'method':'asc','k':k, 'theta':theta,'max':max,
                              'SSE':SSE_a, 'Total_time':total_time_a, 'allSSEs': all_SSEs}
                
                print(asc_current)
                # results= results.append(asc_current, ignore_index=True)
                # all_results= all_results.append(asc_current, ignore_index=True)
                if results.empty or results.shape[0] == 0:
                    results = pd.DataFrame([asc_current])
                    results = pd.DataFrame([asc_current])
                else:
                    results = pd.concat([results, pd.DataFrame([asc_current])], ignore_index=True)
                
                if all_results.empty or all_results.shape[0] == 0:
                    all_results = pd.DataFrame([asc_current])
                else:
                    all_results = pd.concat([all_results, pd.DataFrame([asc_current])], ignore_index=True)


        
        results[columns].to_csv(input_path+f'/outs3/{dataset_name}_exp3.csv')
        
        
    all_results[columns].to_csv(input_path+f'/outs3/exp3_2.csv')   
    print('Finished.')
if __name__ == "__main__":
    __main__(sys.argv)  