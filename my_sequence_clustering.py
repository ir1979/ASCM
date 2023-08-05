import numpy as np

""" Clusters sequential array `data` into `k` optimal groups
    This function is based on paper "On the optimal clustering of sequential data", SIAM 2002
"""
def accelerated_sequence_clustering_basic(data, k):
    # convert to numpy array
    data = np.array(data)

    
    # check dimensions
    if data.ndim<1 or data.ndim>2:
        print('Error in input data')
        exit(-1)

    # expand dim of data if it is one dimensional
    if data.ndim==1:
        data = data[:, None]
    
    n, m = data.shape
    
    left = -np.ones((k+1, n+1))
    cntr = np.zeros((k+1, n+1, m))
    size = np.ones((k+1, n+1), np.integer)

    # for one cluster
    for j in range(1, n+1):
        if j==1:
            left[1][1] = SSE = 0
            cntr[1][1] = m = data[0]
        else:
            m = (m_old*(j-1)+data[j-1])/j
            cntr[1][j] = m;
            
            # SSE is updated incrementally
            SSE += (data[j-1] - m) @ (data[j-1] - m_old)
            left[1][j] = SSE

        size[1][j] = j
        m_old = m

    for  i in range(2, k+1):
        for j in range(n, i-1, -1):
            curCost=0.0
            top_sz = j-i+1

            for sz in range(1, top_sz+1):
                if sz==1:
                    m = data[j-sz]
                    SSE = 0
                    curCost = left[i-1][j-sz]
                else:                    
                    m = (m_old*(sz-1)+data[j-sz])/sz
                    SSE += (data[j-sz]-m_old)@(data[j-sz]-m)
                    curCost = left[i-1][j-sz] + SSE

                if curCost < left[i][j] or left[i][j]<0:
                    cntr[i][j] = m
                    left[i][j] = curCost
                    size[i][j] = sz
                                
                m_old = m;

    i = k
    j = n

    sizes = []
    centers = []
    while i>0:
        cur_sz = size[i][j]
        sizes = [cur_sz] + sizes
        centers = [cntr[i][j]] + centers

        i-=1
        j-=cur_sz;

    final_SSE = left[k][n]
    return centers, sizes, final_SSE, size, left



if __name__ == '__main__':
    a = [1,2,3,5,6]
    centers, sizes, final_SSE, size, left = accelerated_sequence_clustering_basic(a, 2)
    print(centers, sizes, final_SSE, size, left)

    print()
    b = np.random.random_integers(0,10, size=(10,4))
    centers, sizes, final_SSE, size, left = accelerated_sequence_clustering_basic(b, 3)
    print(centers, sizes, final_SSE, size, left)



