import numpy as np
import matplotlib.pyplot as plt
import accelerated_sequence_clustering as ascm

# read data from complex3031.txt
dataxy = np.loadtxt("aggregation788_2.csv", delimiter=',')

# extract and remove the last column
classes = dataxy[:,-1]
dataxy = dataxy[:,0:-1]

sort_idx = np.argsort(classes)
dataxy = dataxy[sort_idx, :]

# find the start idx to be the closest point to the center
center = np.mean(dataxy, axis=0)
distances = np.linalg.norm(dataxy - center, axis=1)
start_idx = np.argmin(distances)


# change bast_start_idx to -1 to search for the best start_idx
# change number of wanted clusters
best_start_idx = 0
wanted_clusters = 7
best_SSE = np.inf

# now start_idx is found (or preset) and is used to construct the sequence
        

tour = np.arange(len(dataxy))
plt.plot(dataxy[tour,0], dataxy[tour,1])
plt.savefig('outputs/aggregation788.png', dpi=300)
plt.savefig('outputs/aggregation788.pdf')
plt.show()


ordered_data = dataxy[tour, :]

sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, 20, 10, 20, False)

# print(all_SSEs)
plt.figure(0)
plt.plot(range(1,len(all_SSEs)+1), all_SSEs)
plt.xticks(range(1,len(all_SSEs)+1))
plt.savefig('outputs/all_SSEs.png', dpi=300)
plt.show()


for selected_k in range(wanted_clusters, wanted_clusters+1):
    # select k based on elbow point selection
    #selected_k = 10
    sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, selected_k, 10, 20, False)
    cum_sizes = np.cumsum(sizes)
    start = 0

    plt.clf()
    plt.figure(selected_k)
    markers = ["." , "x" , "<" , "v" , "^" , "o", ">"]
    colors = ['c','g','b','y','m', 'r', 'k']
    
    for i in range(len(sizes)):
        # plt.plot(ordered_data[:, 0], ordered_data[:, 1], color = 'lightgray', linewidth=0.5)
        plt.plot(ordered_data[start:(cum_sizes[i]-1), 0], ordered_data[start:(cum_sizes[i]-1), 1], marker=markers[i%7], color=colors[i%7], linewidth=0.75)
        start = cum_sizes[i]
    plt.title('k = ' + str(selected_k) + ', SSE = ' + str(SSE))
    plt.savefig('outputs/k' + str(selected_k) + '.png', dpi=300)
    plt.savefig('outputs/k' + str(selected_k) + '.pdf')
    plt.close()
        
print('Finished.')

