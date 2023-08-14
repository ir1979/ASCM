import numpy as np
import matplotlib.pyplot as plt
import accelerated_sequence_clustering as ascm


# Provide the path to your TSPLIB data file and the tour as a list of city indices

# Greece
tsplib_file = "datasets_tsp/gr9882.tsp"
tour_file = "datasets_tsp/gr9882.tour"

# # Japan
# tsplib_file = "datasets_tsp/ja9847.tsp"
# tour_file = "datasets_tsp/ja9847.tour"

# # China
# tsplib_file = "datasets_tsp/ch71009.tsp"
# tour_file = "datasets_tsp/ch71009.tour"

k_MAX = 10
MAX_STALL = 5
THETA = 3
VERBOSE = True

def read_tsplib_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coordinates_section = False
    nodes = {}

    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            coordinates_section = True
        elif coordinates_section:
            if line.strip() == "EOF":
                break
            parts = line.strip().split()
            node_id = int(parts[0])
            y = float(parts[1])
            x = float(parts[2])
            nodes[node_id] = (x, y)

    return nodes

def read_tsplib_tour(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tour = []
    flag = False

    for line in lines:
        if line.startswith("TOUR_SECTION"):
            flag = True
            continue
        
        if flag:
            city_id = line.strip()
            if city_id == 'EOF' or city_id == '-1':
                break
            tour.append(int(city_id))

    return tour

# rotate tour starting from the most distant city from previous one in the original tour
def modify_tour_old(nodes, tour):
    x_values = [node[0] for node in nodes.values()]
    y_values = [node[1] for node in nodes.values()]

    tour_len = len(tour)
    tour_modified = []
    
    max_dist = 0
    start_idx = 0

    for i in range(tour_len):
        start = tour[i] - 1
        end = tour[(i + 1) % tour_len] - 1
        dist = np.sqrt((x_values[start] - x_values[end])**2 + (y_values[start] - y_values[end])**2)
        if dist>max_dist:
            max_dist = dist
            start_idx = (i + 1) % tour_len

    # rotate left tour list by start_idx
    if start_idx>0:
        tour_modified = tour[start_idx:] + tour[:start_idx]
    else:
        tour_modified = tour
    
    return tour_modified

# rotate tour starting from the most distant city from center
def modify_tour(nodes, tour):
    x_values = [node[0] for node in nodes.values()]
    y_values = [node[1] for node in nodes.values()]

    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    tour_len = len(tour)
    tour_modified = []
    
    max_dist = 0
    start_idx = 0

    for i in range(tour_len):
        start = tour[i] - 1
        dist = np.sqrt((x_values[start] - x_mean)**2 + (y_values[start] - y_mean)**2)
        if dist>max_dist:
            max_dist = dist
            start_idx = i

    # rotate left tour list by start_idx
    if start_idx>0:
        tour_modified = tour[start_idx:] + tour[:start_idx]
    else:
        tour_modified = tour
    
    return tour_modified


def find_elbow_point(data):
    # Compute the cumulative sum of squared differences from the mean
    cumulative_variances = np.cumsum((data - np.mean(data)) ** 2)
    
    # Normalize the cumulative variances to [0, 1]
    normalized_variances = cumulative_variances / cumulative_variances[-1]
    
    # Find the index of the "knee" point using the second derivative
    diff_normalized_variances = np.diff(normalized_variances)
    knee_index = np.argmin(diff_normalized_variances)+1
    
    return knee_index, diff_normalized_variances

def generate_distinct_colors(n):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n)]
    return colors

def plot_tsp_solution(nodes):
    seq_data = nodes

    size, SSE, all_SSE, _ = ascm.accelerated_sequence_clustering_approximated3_2d(seq_data, k_MAX, MAX_STALL, THETA, VERBOSE)
    all_SSE = np.array(all_SSE)
    # print(all_SSE)

    elbow_idx, diff_normalized_variances = find_elbow_point(all_SSE)

    k_selected = int(elbow_idx) + 1
    SSE_selected = all_SSE[k_selected-1]
    print('Selected k:   ', k_selected)
    print('Selectec SSE: ', SSE_selected)  

    # plot all_SSE and diff_normalized_variances (for debug only)
    # fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    # axs[0].plot(range(1, len(all_SSE)+1), all_SSE, marker='o')
    # axs[1].plot(range(2, len(all_SSE)+1), diff_normalized_variances, marker='x')

    plt.plot(range(1, len(all_SSE)+1), all_SSE, marker='o')
    plt.annotate('The selected $k$: ' + str(k_selected),
                xy=(k_selected, SSE_selected),
                xytext=(k_selected+3, SSE_selected*1.5),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10,
                color='red')

    plt.xticks(range(1, len(all_SSE)+2))
    plt.xlabel("The number of clusters $k$")
    plt.ylabel("$SSE$")
    plt.show()

    # to compute size_selected we have to call the method again
    size_selected, _, _, _ = ascm.accelerated_sequence_clustering_approximated3_2d(seq_data, k_selected, MAX_STALL, THETA, VERBOSE)

    cur_cluster_idx = 0
    cumsum_size = np.cumsum(size_selected)
    colors = generate_distinct_colors(k_selected)

    plt.figure(figsize=(8, 6))
    for i in range(nodes.shape[0] - 1):
        start = i
        end = i+1
        if i==cumsum_size[cur_cluster_idx]:
            cur_cluster_idx += 1
        plt.plot([nodes[start,0], nodes[end,0]], [nodes[start,1], nodes[end,1]], color=colors[cur_cluster_idx])

    start_idx = 0 
    size_selected_list = list(size_selected)

    # Create data for the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 0.01 * (np.max(nodes[:,0])-np.min(nodes[:,0]))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    for i in range(len(size_selected_list)):
        end_idx = start_idx + size_selected_list[i]
        x_mean = np.mean(nodes[start_idx:end_idx,0])
        y_mean = np.mean(nodes[start_idx:end_idx,1])
        plt.plot(x_mean, y_mean, color=colors[i], marker='X', markersize=10)
        plt.plot(x_mean+x, y_mean+y, color='black')
        start_idx = end_idx
                         

    # Connect the last and first cities to complete the tour
    # start = tour[-1] - 1
    # end = tour[0] - 1
    # plt.plot([x_values[start], x_values[end]], [y_values[start], y_values[end]], color='red')

    plt.title("SSE = " + str(SSE_selected) + ", k = " + str(k_selected))
    # plt.xlabel("X-coordinate")
    # plt.ylabel("Y-coordinate")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()

    # Maximize the window to make it fullscreen
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":

    # Read coordinates from the TSPLIB data file
    coordinates = read_tsplib_file(tsplib_file)

    # Read TSP tour solution from a *.tour file


    # Read tour from a *.tour file
    tsp_solution_tour = read_tsplib_tour(tour_file)  # Example tour, replace with your own

    tsp_solution_tour_modified = modify_tour(coordinates, tsp_solution_tour)

    x_values = [node[0] for node in coordinates.values()]
    y_values = [node[1] for node in coordinates.values()]

    # sequential data clustering
    # vertical concat of x_values and y_values into a numpy array
    nodes = (np.vstack((np.array(x_values).T, np.array(y_values).T)).T)[np.array(tsp_solution_tour_modified)-1]
    x_values = nodes[:,0]
    y_values = nodes[:,1]

    # Plot the TSPLIB data and TSP solution
    plot_tsp_solution(nodes)

    print('Finished.')
