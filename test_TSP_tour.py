import matplotlib.pyplot as plt

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

def plot_tsplib_data(nodes):
    x_values = [node[0] for node in nodes.values()]
    y_values = [node[1] for node in nodes.values()]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c='b', marker='.')

    # for node_id, (x, y) in nodes.items():
    #     plt.annotate(str(node_id), (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('TSPLIB Data File Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    tsplib_file = "datasets_tsp/gr9882.tsp"
    nodes = read_tsplib_file(tsplib_file)
    plot_tsplib_data(nodes)