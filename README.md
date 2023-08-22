This is a sequential clustering module.


## Install from PyPI:
```bash
pip install accelerated-sequence-clustering
```

## Usage:
```py 
import accelerated_sequence_clustering as ascm
import numpy as np

np.random.seed(1)
a_2d = np.random.randint(0, 21, (20, 2))

# basic sequetial data clustering 
# reported in """On the accelerated clustering of sequential data", SIAM 2002"""
k = 3
verbose = True
sizes, SSE, _, _ = ascm.basic_sequence_clustering_2d(a_2d, k, verbose)
print(str(k), " ", str(sizes), " ", str(SSE))

k = 3
max_stall = 3
theta = 15
verbose = True
sizes, SSE, all_SSEs, total_saved_operations = ascm.accelerated_sequence_clustering_approximated3_2d(a_2d, k, max_stall, theta, verbose)
print(str(k), " ", str(sizes), " ", str(SSE))
```

```bash
$ python -m project_name
```
