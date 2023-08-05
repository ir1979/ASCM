import numpy as np
import time
import matplotlib.pyplot as plt

import accelerated_sequence_clustering as ascm

import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1)
a = np.random.randint(0, 21, (20)).astype(np.double)
a_2d = np.random.randint(0, 21, (20, 2))

#################
print('** Basic Sequence clustering of a List: ')
b = a.tolist()
print(b[:5])
print(f'len(b)={len(a)}, type(b)={type(b)}')
start = time.time()
cntrs, sizes, SSE, _, ll = ascm.basic_sequence_clustering(b, 3)
end = time.time()

print(cntrs)
print(sizes)
print(SSE)

print(end-start, ' seconds.')

#################

print('** Basic Sequence clustering of a Numpy array: ')
print(a[:5])
print(f'len(a)={len(a)}, type(a)={type(a)}')

start = time.time()
cntrs, sizes, SSE, _, _ = ascm.basic_sequence_clustering(a, 3)
end = time.time()

print(cntrs)
print(sizes)
print(SSE)

print(end-start, ' seconds.')


#################
print()
print('** Accelerated Sequence clustering of a Numpy array: ')
print(a[:5])
print(f'len(a)={len(a)}, type(a)={type(a)}')

start = time.time()
cntrs, sizes, SSE, _, _ = ascm.accelerated_sequence_clustering_approximated(a, 3, 1, 1)
end = time.time()

print(cntrs)
print(sizes)
print(SSE)

print(end-start, ' seconds.')



######################

# print('** Numpy range of k values between 3 and 5: ')
# print(a[:5])
# print(f'len(a)={len(a)}, type(a)={type(a)}')

# k_min = 3
# k_max = 5

# start = time.time()
# for k in range(k_min, k_max+1):
#     cntrs, sizes, SSE, _, _ = ascm.basic_sequence_clustering(a, k)
# end = time.time()
# print(end-start, ' seconds.')

# start = time.time()
# cntrs, sizes, SSE, _, _ = ascm.basic_sequence_clustering(a, k_min, k_max)
# end = time.time()

# for cur_k, cur_cntrs, cur_sizes, cur_SSE in zip(range(k_min, k_max+1), cntrs, sizes, SSE): 
#     print(cur_k)
#     print(cur_cntrs)
#     print(cur_sizes)
#     print(cur_SSE)


# print(end-start, ' seconds.')

######################

print('** Basic Sequence clustering of a 2d Numpy array for k between 3 and 5: ')
print(a_2d[:5])
print(f'len(a_2d)={len(a_2d)}, type(a_2d)={type(a_2d)}')

k_min = 5
k_max = 5

start = time.time()
for k in range(k_min, k_max+1):
    sizes, SSE, _, _ = ascm.basic_sequence_clustering_2d(a_2d, k)
    print(str(k), " ", str(sizes), " ", str(SSE))
end = time.time()
print(end-start, ' seconds.')

# start = time.time()
# sizes, SSE, _, _ = ascm.basic_sequence_clustering_2d(a_2d, k_min, k_max)
# end = time.time()

# for cur_k, cur_sizes, cur_SSE in zip(range(k_min, k_max+1), sizes, SSE): 
#     print(cur_k)
#     print(cur_sizes)
#     print(cur_SSE)


# print(end-start, ' seconds.')


start = time.time()
for k in range(k_min, k_max+1):
    sizes, SSE = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, 5, 3, 0)
    print(str(k), " ", str(sizes), " ", str(SSE))
end = time.time()
print(end-start, ' seconds.')

print('Finished.')


