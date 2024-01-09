import time
import accelerated_sequence_clustering as ascm
import numpy as np

def process_theta(dataset_name, a_2d, k, max, theta, SSE_b, all_SSEsBasic, total_time_b):
    print(f'dataset_name={dataset_name}, k={k}, max={max}, theta={theta}')

    start_a = time.time()
    sizes, SSE_a, all_SSEsAccelerated, _ = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, max, theta, False)
    end_a = time.time()
    total_time_a = end_a - start_a
    speed_up=total_time_b/total_time_a
    Gap=(SSE_a-SSE_b)/SSE_b*100
    all_SSEsAccelerated = np.array(all_SSEsAccelerated)

    asc_current = {
        'dataset_name': dataset_name,
        'method': 'asc',
        'k': k,
        'theta': theta,
        'max': max,
        'SSE': SSE_a,
        'Total_time': total_time_a,
        'allSSEsBasic': all_SSEsBasic, 
        'allSSEsAccelerated': all_SSEsAccelerated,
        'speed_up': speed_up,
        'Gap': Gap
    }

    return asc_current