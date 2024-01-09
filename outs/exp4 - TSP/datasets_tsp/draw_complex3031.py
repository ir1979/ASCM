import numpy as np
import matplotlib.pyplot as plt
import accelerated_sequence_clustering as ascm

# read data from complex3031.txt
dataxy = np.loadtxt("complex3031.csv")

# remove first column
dataxy = dataxy[:,1:]

# find the most distant record from the center
center = np.mean(dataxy, axis=0)
distances = np.linalg.norm(dataxy - center, axis=1)
most_distant_idx = np.argmax(distances)

tour = np.loadtxt("complex3031.tour", dtype=int)

# find most_distant_idx in tour
most_distant_idx_in_tour = np.where(tour == most_distant_idx)[0][0]

tour = np.roll(tour, -most_distant_idx_in_tour)

plt.plot(dataxy[tour,0], dataxy[tour,1])
plt.show()

np.savetxt('complex3031.in_tsp_order.txt', dataxy[tour, :])

ordered_data = dataxy[tour, :]

sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, 20, 10, 20, False)

print(all_SSEs)
plt.plot(range(1,len(all_SSEs)+1), all_SSEs)
plt.xticks(range(1,len(all_SSEs)+1))
plt.show()


for selected_k in range(1, 21):
    # select k based on elbow point selection
    #selected_k = 10
    sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(ordered_data, selected_k, 10, 20, False)
    cum_sizes = np.cumsum(sizes)-1
    start = 0


    plt.figure(selected_k)
    for i in range(len(sizes)):
        plt.plot(ordered_data[start:cum_sizes[i], 0], ordered_data[start:cum_sizes[i], 1])
        start = cum_sizes[i]
    plt.title('k = ' + str(selected_k) + ', SSE = ' + str(SSE))
    plt.savefig('outputs/k' + str(selected_k) + '.png')
    
#plt.show()

print('Finished.')

