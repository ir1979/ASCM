# pip install python-tsp
# see https://pypi.org/project/python-tsp/

import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search

data = np.loadtxt("complex9.txt", delimiter=',')

distance_matrix = euclidean_distance_matrix(data)

permutation, distance = solve_tsp_local_search(distance_matrix)
print(permutation, distance)
