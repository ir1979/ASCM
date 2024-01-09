import numpy as np
import matplotlib.pyplot as plt
import accelerated_sequence_clustering as ascm

# read data from complex3031.txt
dataxy = np.loadtxt("aggregation788.csv")

# remove first column
dataxy = dataxy[:,1:]

# find the start idx
center = np.mean(dataxy, axis=0)
distances = np.linalg.norm(dataxy - center, axis=1)
start_idx = np.argmin(distances)


# change bast_start_idx to -1 to search for the best start_idx
# change number of wanted clusters
best_start_idx = -1
wanted_clusters = 7
best_SSE = np.inf
orig_tour = np.loadtxt("aggregation788.tour", dtype=int)

if best_start_idx == -1:
    for start_idx in range(len(dataxy)):
        print(str(start_idx) + "...", end='')
        tour = orig_tour

        # find start_idx in tour
        start_idx_in_tour = np.where(tour == start_idx)[0][0]

        tour = np.roll(tour, -start_idx_in_tour)

        ordered_data = dataxy[tour, :]

        for selected_k in range(wanted_clusters, wanted_clusters+1):
            # select k based on elbow point selection
            #selected_k = 10
            sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, selected_k, 10, 20, False)
            cum_sizes = np.cumsum(sizes)-1
            start = 0

            if SSE < best_SSE:
                best_start_idx = start_idx
                best_SSE = SSE
                print('Best SSE: ' + str(best_SSE) + ' at start_idx: ' + str(best_start_idx))

        print('Ok.')



# now start_idx is found (or preset) and is used to construct the sequence
        
print('Selected start_idx: ' + str(best_start_idx))
tour = np.loadtxt("aggregation788.tour", dtype=int)

# find start_idx in tour
start_idx_in_tour = np.where(tour == start_idx)[0][0]

tour = np.roll(tour, -start_idx_in_tour)

plt.plot(dataxy[tour,0], dataxy[tour,1])
plt.show()

np.savetxt('aggregation788.in_tsp_order.txt', dataxy[tour, :])

ordered_data = dataxy[tour, :]

sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, 20, 10, 20, False)

# print(all_SSEs)
plt.figure(0)
plt.plot(range(1,len(all_SSEs)+1), all_SSEs)
plt.xticks(range(1,len(all_SSEs)+1))
plt.savefig('outputs/all_SSEs.png')
plt.show()


for selected_k in range(wanted_clusters, wanted_clusters+1):
    # select k based on elbow point selection
    #selected_k = 10
    sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, selected_k, 10, 20, False)
    cum_sizes = np.cumsum(sizes)-1
    start = 0

    plt.figure(selected_k)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    for i in range(len(sizes)):
        plt.plot(ordered_data[:, 0], ordered_data[:, 1], color = 'lightgray', linewidth=0.5)
        plt.plot(ordered_data[start:cum_sizes[i], 0], ordered_data[start:cum_sizes[i], 1], marker=markers[i%7], color=colors[i%7], linewidth=0.75)
        if selected_k == 1:
            plt.plot(center[0], center[1], 'x', markersize=10, color='k')
            plt.plot(ordered_data[start, 0], ordered_data[start, 1], 'x', markersize=10, color='k')
        start = cum_sizes[i]
    plt.title('k = ' + str(selected_k) + ', SSE = ' + str(SSE))
    plt.savefig('outputs/k' + str(selected_k) + '.png')
    plt.close()
        
print('Finished.')

