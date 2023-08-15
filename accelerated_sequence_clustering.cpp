#include <iostream>
#include <vector>
#include <float.h>
#include "accelerated_sequence_clustering.h"
#include <math.h>

// using namespace std;

#define VERBOSE
// #define VERBOSE2

#define pow2(x) ((x)*(x))

double compute_center(std::vector<double> data, int en)
{
    int st=0;
    double sum = 0;
    for (int i=st;i<en;i++)
        sum+= data[i];
    
    return sum/(en-st);
}

double *compute_center_2d(const double *data, int nrows, int ncols)
{
    double *res = new double[ncols]();
    int counter = 0;
    for (int i=0;i<nrows;i++)
        for (int j=0;j<ncols;j++)
            res[j] += data[counter++];
    
    for (int j=0;j<ncols;j++)
        res[j] /= nrows;

    return res;
}

double *zero_mean_2d(const double *seq_data, int nrows, int ncols)
{
    double *mean = compute_center_2d(seq_data, nrows, ncols);
    double *data = new double[nrows*ncols];

    int counter = 0;
    for (int i=0;i<nrows; i++)
        for (int j=0;j<ncols;j++, counter++)
            data[counter] = seq_data[counter] - mean[j];

    return data;
}


double compute_center(std::vector<double> data, int st, int en)
{
    double sum = 0;
    for (int i=st;i<en;i++)
        sum+= data[i];
    
    return sum/(en-st);
}


double compute_SSE(std::vector<double> data, int st, int en, double m)
{
    double result = 0;

    for (int c=st; c<en; c++)
        result += pow2(data[c]-m);
    
    return result;
}

double compute_SSE(std::vector<double> data, int st, int en)
{
    double m = compute_center(data, st, en);
    double result = 0;

    for (int c=st; c<en; c++)
        result += pow2(data[c]-m);
    
    return result;
}

double compute_SSE(std::vector<double> data, int en, double m)
{
    return compute_SSE(data, 0, en, m);
}


double compute_SSE(std::vector<double> data, int en)
{
    return compute_SSE(data, 0, en);
}



// for simple python lists
void accelerated_sequence_clustering_approximated(std::vector<double> data, 
        int k, 
        int MAX_STALL,
        double theta,
        std::vector<double>& out_cntrs, 
        std::vector<int>& out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left)
{
    // for debug
    // std::cout << "The list version" << std::endl;

    accelerated_sequence_clustering_approximated((int)data.size(), &data[0], k, MAX_STALL, theta, out_cntrs, out_sizes, out_SSE, out_internal_size, out_internal_left);

}


// for simple python lists
// uses flat data structures instead of vecvec
void accelerated_sequence_clustering_approximated2(std::vector<double> data, 
        int k, 
        int MAX_STALL,
        double theta,
        std::vector<double>& out_cntrs, 
        std::vector<unsigned int>& out_sizes,
        double & out_SSE,
        std::vector<unsigned int> & out_internal_size_flat,
        std::vector<double> & out_internal_left_flat)
{
    // for debug
    // std::cout << "The list version" << std::endl;

    accelerated_sequence_clustering_approximated2((int)data.size(), &data[0], k, MAX_STALL, theta, out_cntrs, out_sizes, out_SSE, out_internal_size_flat, out_internal_left_flat);

}


// for simple python lists
// uses flat data structures instead of vecvec
// space saving
void accelerated_sequence_clustering_approximated3(std::vector<double> data, 
        int k, 
        int MAX_STALL,
        double theta,
        std::vector<int>& out_sizes,
        double & out_SSE)
{
    accelerated_sequence_clustering_approximated3((int)data.size(), &data[0], k, MAX_STALL, theta, out_sizes, out_SSE);
}


// for simple python lists
void accelerated_sequence_clustering(std::vector<double> data, 
        int k, 
        std::vector<int>& out_sizes,
        double & out_SSE)
{
    // for debug
    // std::cout << "The list version" << std::endl;

    accelerated_sequence_clustering((int)data.size(), &data[0], k, out_sizes, out_SSE);

}


// for simple python lists
void basic_sequence_clustering(std::vector<double> data, 
        int k, 
        std::vector<double>& out_cntrs, 
        std::vector<int>& out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left)
{
    // for debug
    // std::cout << "The list version" << std::endl;

    basic_sequence_clustering((int)data.size(), &data[0], k, out_cntrs, out_sizes, out_SSE, out_internal_size, out_internal_left);
}

// a range of k values between k_min and k_max
// for simple python lists
void basic_sequence_clustering(std::vector<double> data, 
        int k_min,
        int k_max, 
        std::vector<std::vector<double> > & out_cntrs2, 
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2,
        std::vector<std::vector<int> > & out_internal_size2,
        std::vector<std::vector<double> > & out_internal_left2)
{
    // for debug
    // std::cout << "The list version" << std::endl;

    basic_sequence_clustering(
        (int)data.size(), 
        &data[0], k_min, k_max, out_cntrs2, out_sizes2, out_SSE2, 
        out_internal_size2, 
        out_internal_left2);

}


// approximated
// for numpy arrays
void accelerated_sequence_clustering_approximated(int n, 
        double *data, 
        int k,
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left
        )
{
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""
    
    std::vector<std::vector<double> > left(k+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<double> > cntr(k+1, std::vector<double>(n+1, 0));
    std::vector<std::vector<int> >    size(k+1, std::vector<int>(n+1, -1));
    

    double m, m_old, SSE;

    // for j = 1
    left[1][1] = SSE = 0;
    cntr[1][1] = m = data[0];


    size[1][1] = 1;
    m_old = m;

    for (int j=2; j<n+1; j++)
    {
        m = (m_old*(j-1)+data[j-1])/j;
        cntr[1][j] = m;
        
        SSE += (data[j-1] - m) * (data[j-1] - m_old);
        left[1][j] = SSE;

        size[1][j] = j;
        m_old = m;
    }

    // std::cout << "The new Version." << std::endl;

    int top_sz = n-1;
    double curCost;
    
    // double threshold = 0;
    for (int i=2;i<k+1; i++)
    {
        double lambda = 0;
        for (int j=n; j>=i; j--)
        {
            curCost=0;
            int jp1 = j+1;
            top_sz = jp1 - i;

            if (j<n && size[i][jp1]>1) 
            {
                cntr[i][j] = (cntr[i][jp1]*size[i][jp1]-data[j])/(size[i][jp1]-1);
                left[i][j] = left[i][jp1] - ((data[j]-cntr[i][jp1])*(data[j]-cntr[i][j]));
                size[i][j] = size[i][jp1]-1;
            }

            if (lambda==0 || data[j]*data[j] >= lambda*theta)
            {

                int sz = 1;
                m = data[j-sz];
                SSE = 0;
                curCost = left[i-1][j-1];

                // update if necessary
                if (curCost < left[i][j] || left[i][j]<0)
                {
                    cntr[i][j] = m;
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                m_old = m;

                int jmsz = j - sz;
                for (int sz=2; sz<=top_sz; sz++)
                {
                    jmsz--; // = j - sz;
                    m = (m_old*(sz-1)+data[jmsz])/sz;
                    SSE += (data[jmsz]-m_old)*(data[jmsz]-m);

                    curCost = left[i-1][jmsz] + SSE; //compute_SSE(data, j-sz, j, m);

                    // update if necessary
                    if (curCost < left[i][j] || left[i][j]<0)
                    {
                        cntr[i][j] = m;
                        left[i][j] = curCost;
                        size[i][j] = sz;
                    }

                    m_old = m;
                }
            }
            else 
            {
                // imporves number of updates by initialization based on the results of the previous step
                // if (j<n && size[i][j+1]>1) 
                // {
                //     cntr[i][j] = (cntr[i][j+1]*size[i][j+1]-data[j])/(size[i][j+1]-1);
                //     left[i][j] = left[i][j+1] - ((data[j]-cntr[i][j+1])*(data[j]-cntr[i][j]));
                //     size[i][j] = size[i][j+1]-1;
                // }
                
                int stall = 0;
                int sz = 1;
                m = data[j-sz];
                SSE = 0;
                curCost = left[i-1][j-1];
                if (curCost < left[i][j] || left[i][j]<0)
                {
                    cntr[i][j] = m;
                    left[i][j] = curCost;
                    size[i][j] = sz;
                    stall = 0;
                }
                else 
                {
                    stall++;
                }
                m_old = m;
                int jmsz = j - sz;

                for (int sz=2; sz<=top_sz; sz++)
                {
                    jmsz--;   // j - sz
                    m = (m_old*(sz-1)+data[jmsz])/sz;
                    SSE += (data[jmsz]-m_old)*(data[jmsz]-m);

                    curCost = left[i-1][jmsz] + SSE; //compute_SSE(data, j-sz, j, m);

                    // update if necessary
                    if (curCost < left[i][j] || left[i][j]<0)
                    {
                        cntr[i][j] = m;
                        left[i][j] = curCost;
                        size[i][j] = sz;
                        stall = 0;
                    }
                    else 
                    {
                        stall++;

                        if (stall>=MAX_STALL)// && (SSE-left[i-1][j-sz])>=threshold) 
                        {
                            break;
                        }
                    }
                    m_old = m;
                }
            }

            if (size[i][j]>1)
            {
                double tmp = data[j-1]-cntr[i][j];
                lambda =  tmp*tmp*size[i][j]/(size[i][j]-1);
            }
            else 
            {
                lambda = 0;

            }
        } // end of for j
    }  // end of for i

    //std::cout << "total updates: " << total_updates << std::endl;

    int i = k;
    int j = n;

    std::vector<double> centers;
    std::vector<int> sizes; 

    while(i>0)
    {
        int cur_sz = size[i][j];

        sizes.insert(sizes.begin(), cur_sz);
        centers.insert(centers.begin(), cntr[i][j]);

        i--;
        j-=cur_sz;
    }

    out_cntrs = centers;
    out_sizes = sizes;
    out_SSE = left[k][n];
    
    out_internal_size = size;
    out_internal_left = left;
}


// approximated
// for numpy arrays
// using flat vectors
void accelerated_sequence_clustering_approximated2(int n, 
        double *data, 
        int k,
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<unsigned int> & out_sizes_unsigned,
        double & out_SSE,
        std::vector<unsigned int> & out_internal_size_flat,
        std::vector<double> & out_internal_left_flat
        )
{    
    unsigned int kp1mnp1 = (k+1)*(n+1);

    std::vector<double> left(kp1mnp1, DBL_MAX);
    std::vector<double> cntr(kp1mnp1, 0);
    std::vector<unsigned int> size(kp1mnp1, 0);
    

    double m, m_old, SSE;

    // for j = 1
    left[n+2] = SSE = 0;   // 1*(n+1)+1
    cntr[n+2] = m = data[0];
    size[n+2] = 1;

    m_old = m;

    for (unsigned int j=2; j<(unsigned int)(n+1); j++)
    {
        m = (m_old*(j-1)+data[j-1])/j;
        cntr[n+1+j] = m;    // 1*(n+1)+j
        
        SSE += (data[j-1] - m) * (data[j-1] - m_old);
        left[n+1+j] = SSE;    // 1*(n+1)+j

        size[n+1+j] = j;    // 1*(n+1)+j
        m_old = m;
    }

    // std::cout << "The new Version." << std::endl;

    unsigned int top_sz = n-1;
    double curCost;
    
    // double threshold = 0;
    for (unsigned int i=2;i<(unsigned int)(k+1); i++)
    {
        unsigned int imnp1 = i*(n+1);
        double lambda = 0;
        for (unsigned int j=n; j>=i; j--)
        {
            curCost=0;
            unsigned int jp1 = j+1;
            unsigned int imnp1pj = imnp1+j;
            unsigned int imnp1pjp1 = imnp1 + jp1;  
            top_sz = jp1 - i;

            if (j<(unsigned int)(n) && size[imnp1pjp1]>1) 
            {
                cntr[imnp1pj] = (cntr[imnp1pjp1]*size[imnp1pjp1]-data[j])/(size[imnp1pjp1]-1);
                left[imnp1pj] = left[imnp1pjp1] - ((data[j]-cntr[imnp1pjp1])*(data[j]-cntr[imnp1pj]));
                size[imnp1pj] = size[imnp1pjp1]-1;
            }

            if (lambda==0 || data[j]*data[j] >= lambda*theta)
            {

                int sz = 1;
                m = data[j-sz];
                SSE = 0;
                curCost = left[(i-1)*(n+1)+j-1];

                // update if necessary
                if (curCost < left[imnp1pj])
                {
                    cntr[imnp1pj] = m;
                    left[imnp1pj] = curCost;
                    size[imnp1pj] = sz;
                }

                m_old = m;

                int jmsz = j - sz;
                for (unsigned int sz=2; sz<=top_sz; sz++)
                {
                    jmsz--; // = j - sz;
                    m = (m_old*(sz-1)+data[jmsz])/sz;
                    SSE += (data[jmsz]-m_old)*(data[jmsz]-m);

                    curCost = left[(i-1)*(n+1)+jmsz] + SSE; //compute_SSE(data, j-sz, j, m);

                    // update if necessary
                    if (curCost < left[imnp1pj])
                    {
                        cntr[imnp1pj] = m;
                        left[imnp1pj] = curCost;
                        size[imnp1pj] = sz;
                    }

                    m_old = m;
                }
            }
            else 
            {
                // imporves number of updates by initialization based on the results of the previous step
                // if (j<n && size[i][j+1]>1) 
                // {
                //     cntr[i][j] = (cntr[i][j+1]*size[i][j+1]-data[j])/(size[i][j+1]-1);
                //     left[i][j] = left[i][j+1] - ((data[j]-cntr[i][j+1])*(data[j]-cntr[i][j]));
                //     size[i][j] = size[i][j+1]-1;
                // }
                
                int stall = 0;
                unsigned int sz = 1;
                m = data[j-sz];
                SSE = 0;
                curCost = left[(i-1)*(n+1)+j-1];
                if (curCost < left[imnp1pj])
                {
                    cntr[imnp1pj] = m;
                    left[imnp1pj] = curCost;
                    size[imnp1pj] = sz;
                    stall = 0;
                }
                else 
                {
                    stall++;
                }
                m_old = m;
                int jmsz = j - sz;

                for (unsigned int sz=2; sz<=top_sz; sz++)
                {
                    jmsz--;   // j - sz
                    m = (m_old*(sz-1)+data[jmsz])/sz;
                    SSE += (data[jmsz]-m_old)*(data[jmsz]-m);

                    curCost = left[(i-1)*(n+1)+jmsz] + SSE; //compute_SSE(data, j-sz, j, m);

                    // update if necessary
                    if (curCost < left[imnp1pj])
                    {
                        cntr[imnp1pj] = m;
                        left[imnp1pj] = curCost;
                        size[imnp1pj] = sz;
                        stall = 0;
                    }
                    else 
                    {
                        stall++;

                        if (stall>=MAX_STALL)// && (SSE-left[i-1][j-sz])>=threshold) 
                        {
                            break;
                        }
                    }
                    m_old = m;
                }
            }

            if (size[imnp1pj]>1)
            {
                double tmp = data[j-1]-cntr[imnp1pj];
                lambda =  tmp*tmp*size[imnp1pj]/(size[imnp1pj]-1);
            }
            else
                lambda = 0;
        } // end of for j
    }  // end of for i

    //std::cout << "total updates: " << total_updates << std::endl;

    int i = k;
    int j = n;

    std::vector<double> centers;
    std::vector<unsigned int> sizes; 

    while(i>0)
    {
        int cur_sz = size[i*(n+1)+j];

        sizes.insert(sizes.begin(), cur_sz);
        centers.insert(centers.begin(), cntr[i*(n+1)+j]);

        i--;
        j-=cur_sz;
    }

    out_cntrs = centers;
    out_sizes_unsigned = sizes;
    out_SSE = left[k*(n+1)+n];
    
    out_internal_size_flat = size;
    out_internal_left_flat = left;
}



// approximated
// for numpy arrays
// space saving
void accelerated_sequence_clustering_approximated3(int nn, 
        double *data, 
        int kk,
        int MAX_STALL,
        double theta,
        std::vector<int> & out_sizes,
        double & out_SSE
        )
{
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""
    
    unsigned n = nn;
    unsigned k = kk;
    unsigned int kn = k*n;

    unsigned long int total_saved_operations = 0;

    //std::vector<unsigned int> size(kn, 0);
    unsigned int *size = new unsigned int[kn]();
    double ** cntr = new double*[k]();
    double ** left = new double*[k]();

    left[0] = new double[n];
    cntr[0] = new double[n];
    
    double m, m_old, sum, SSE;

    // for i = 1
    left[0][0] = SSE = 0;   // 1*(n+1)+1
    cntr[0][0] = m = data[0];
    size[0] = 1;

    sum = m_old = m;

    for (unsigned int j=1; j<n; j++)
    {
        sum += data[j];
        m = sum/(j+1);
        cntr[0][j] = m;    // 1*(n+1)+j
        
        SSE += (data[j] - m) * (data[j] - m_old);
        left[0][j] = SSE;    // 1*(n+1)+j

        size[j] = j+1;    // 1*(n+1)+j
        m_old = m;
    }

    // std::cout << "The new Version." << std::endl;

    unsigned int top_sz = n-1;
    double curCost;
    
    //unsigned int imn, top_sz, sz, jp1, imnpj, imnpjp1, best_size, cur_data, *cur_data_ptr, *prev_left;
    //double lambda, best_left, best_cntr;

    for (unsigned int i=1;i<k; i++)
    {
        double *left_i = left[i] = new double[n];
        for (unsigned int tmp=0;tmp<n;tmp++)
            left_i[tmp] = DBL_MAX;

        double *cntr_i = new double[n]();

        unsigned int imn = i*n;
        double lambda = 0;
        for (unsigned int j=n-1; j>=i; j--)
        {
            curCost=0;
            unsigned int jp1 = j+1;
            unsigned int imnpj = imn+j;
            unsigned int imnpjp1 = imnpj + 1;  
            top_sz = jp1 - i;
            

            if (j<n-1 && size[imnpjp1]>1) 
            {
                size[imnpj] = size[imnpjp1]-1;
                cntr_i[j] = (cntr_i[jp1]*size[imnpjp1]-data[jp1])/size[imnpj];
                left_i[j] = left_i[jp1] - ((data[jp1]-cntr_i[jp1])*(data[jp1]-cntr_i[j]));
            }

            if (lambda==0 || pow2(data[jp1]) >= lambda*theta) //data[jp1]*data[jp1] >= lambda*theta)
            {

                int sz = 1;
                m = data[j]; // [j-sz+1];
                double SSE = 0;
                curCost = left[i-1][j-1];

                // update if necessary
                if (curCost < left_i[j])
                {
                    cntr_i[j] = m;
                    left_i[j] = curCost;
                    size[imnpj] = sz;
                }

                double best_left = left_i[j], best_cntr = cntr_i[j];
                unsigned int best_size = size[imnpj];

                sum = m_old = m;

                //int jmszp1 = j - sz + 1;
                
                double cur_data;

                double *prev_left = &left[i-1][j-2];    // start of sz is 2

                double *cur_data_ptr = &data[j-1];
                for (unsigned int sz=2; sz<=top_sz; sz++)
                {

                    // std::cout << i << j << sz << std::endl;

                    //jmszp1--; 
                    //cur_data = data[jmszp1];
                    cur_data = *cur_data_ptr--;

                    // m = (m_old*(sz-1)+data[jmszp1])/sz;
                    sum += cur_data;
                    m = sum/sz;

                    SSE += (cur_data-m_old)*(cur_data-m);

                    //curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);
                    curCost = *prev_left-- + SSE;

                    // update if necessary
                    if (curCost < best_left) // left[i][j])
                    {
                        best_cntr = m;
                        best_left = curCost;
                        best_size = sz;
                    }

                    m_old = m;
                }
                cntr_i[j] = best_cntr;
                left_i[j] = best_left;
                size[imnpj] = best_size;
            }
            else 
            {
                int stall = 0;
                unsigned int sz = 1;
                m = data[j-sz+1];
                double SSE = 0;
                curCost = left[i-1][j-1];

                if (curCost < left_i[j])
                {
                    cntr_i[j] = m;
                    left_i[j] = curCost;
                    size[imnpj] = sz;
                    stall = 0;
                }
                else 
                {
                    stall++;
                }

                double best_left = left_i[j], best_cntr = cntr_i[j];
                unsigned int best_size = size[imnpj];


                sum = m_old = m;
                //int jmszp1 = j - sz + 1;
                double cur_data;

                double *prev_left = &left[i-1][j-2];    // start of sz is 2
                double *cur_data_ptr = &data[j-1];
                for (unsigned int sz=2; sz<=top_sz; sz++)
                {
                    //jmszp1--;   // j - sz
                    //m = (m_old*(sz-1)+data[jmszp1])/sz;
                    
                    //cur_data = data[jmszp1];
                    cur_data = *cur_data_ptr--;

                    sum += cur_data;
                    m = sum/sz;
                    SSE += (cur_data-m_old)*(cur_data-m);

                    //curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);
                    curCost = *prev_left-- + SSE;

                    // update if necessary
                    if (curCost < best_left) // left[i][j])
                    {
                        best_cntr = m;
                        best_left = curCost;
                        best_size = sz;
                        stall = 0;
                    }
                    else 
                    {
                        stall++;

                        if (stall>=MAX_STALL)// && (SSE-left[i-1][j-sz])>=threshold) 
                        {
                            #ifdef VERBOSE
                                total_saved_operations+=top_sz-sz;
                            #endif
                            #ifdef VERBOSE2
                                std::cout << "i: " << i << ", j: " << j << ", data[jp1]**2: " << data[jp1]*data[jp1] << ", lambda: " << lambda << ", sz: " << sz << ", top_sz:" << top_sz << ", saved: " << top_sz-sz << std::endl;
                            #endif

                            break;
                        }
                    }
                    m_old = m;
                }
                cntr_i[j] = best_cntr;
                left_i[j] = best_left;
                size[imnpj] = best_size;
            }

            if (size[imnpj]>1)
            {
                double tmp = data[j]-cntr_i[j];
                lambda =  tmp*tmp*size[imnpj]/(size[imnpj]-1);
            }
            else if (j>=2)
            {
                lambda = left[i-1][j-1]-left[i-1][j-2];
                //std::cout << "+";
            }
            else // may be imprved later
                lambda = 0;

            // other entries in the last row are not required
            if (i==k-1 && theta>0)
                break;
        } // end of for j

        delete[] left[i-1];
        left[i-1] = 0;
        delete[] cntr[i-1];
        cntr[i-1] = 0;
    }  // end of for i


    //std::cout << "total updates: " << total_updates << std::endl;

    int i = k-1;
    int j = n-1;

    std::vector<int> sizes(k); 

    while(i>=0)
    {
        int cur_sz = (int)size[i*n+j];

        //sizes.insert(sizes.begin(), cur_sz);
        sizes[i] = cur_sz;

        i--;
        j-=cur_sz;
    }

    out_sizes = std::move(sizes);
    out_SSE = left[k-1][n-1];
    
    delete[] left[k-1];
    left[k-1] = 0;
    delete[] cntr[k-1];
    cntr[k-1] = 0;
    delete[] left;
    left = 0;
    delete[] cntr;
    cntr = 0;

    #ifdef VERBOSE
    std::cout << "Total saved operations: " << total_saved_operations << std::endl;
    #endif
}



void cp_dbl(double *dst, const double *src, int d) 
{
    for (int i=0;i<d;i++)
        dst[i] = src[i];
}


void add_dbl(double *dst, const double *src, int d) 
{
    for (int i=0;i<d;i++)
        dst[i] += src[i];
}


void cp_dbl_div(double *dst, const double *src, int n, int d) 
{
    for (int i=0;i<d;i++)
        dst[i] = src[i]/n;
}

double sum_pow2(const double *data, int d)
{
    double res = 0;
    for (int i=0;i<d; i++)
        res += pow2(data[i]);
    
    return res;
}

// approximated
// for numpy arrays
// space saving, i.e., only previous left matrix is preserved
void accelerated_sequence_clustering_approximated3_2d(int nrows,
        int ncols,
        const double *seq_data, 
        int k,
        int MAX_STALL,
        double theta,
        bool verbose,
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<double> & out_All_SSEs,
        long int & total_saved_operations 
        )
{
    //"""This function is an improved version of the paper "On the accelerated clustering of sequential data", SIAM 2002"""
    
    int n = nrows;
    int d = ncols;
    int kn = k*n;

    total_saved_operations = 0;
    std::vector<double> all_SSEs(k); 

    double *data = zero_mean_2d(seq_data, nrows, ncols);

    //std::vector<unsigned int> size(kn, 0);
    double ** left = new double*[k]();
    double ** cntr = new double*[k]();
    int *size = new int[kn]();

    left[0] = new double[n];
    cntr[0] = new double[n*d];
    
    double *m = new double[d], *m_old = new double[d], *sum = new double[d], SSE;

    // for i = 0
    left[0][0] = SSE = 0;   // 1*(n+1)+1
    cp_dbl(m, &data[0], d);
    cp_dbl(&cntr[0][0], m, d);
    size[0] = 1;

    cp_dbl(m_old, m, d);
    cp_dbl(sum, m_old, d);

    for (unsigned int j=1; j<n; j++)
    {
        add_dbl(sum, &data[j*d], d);
        cp_dbl_div(m, sum, j+1, d);
        cp_dbl(&cntr[0][j*d], m, d);    // 1*(n+1)+j
        
        for (int tmp=0;tmp<d;tmp++)
            SSE += (data[j*d+tmp] - m[tmp]) * (data[j*d+tmp] - m_old[tmp]);
        left[0][j] = SSE;    // 1*(n+1)+j

        size[j] = j+1;    // 1*(n+1)+j
        cp_dbl(m_old, m, d);
    }

    // for k=1
    all_SSEs[0] = SSE;

    // std::cout << "The new Version." << std::endl;

    unsigned int top_sz = n-1;
    double curCost;
    
    //unsigned int imn, top_sz, sz, jp1, imnpj, imnpjp1, best_size, cur_data, *cur_data_ptr, *prev_left;
    //double lambda, best_left, best_cntr;

    for (unsigned int i=1;i<k; i++)
    {
        if (verbose)
            std::cout << (double)i/k*100 << "%" << std::endl;

        double *left_i = left[i] = new double[n];
        for (unsigned int tmp=0;tmp<n;tmp++)
            left_i[tmp] = DBL_MAX;

        double *cntr_i = new double[n*d]();

        unsigned int imn = i*n;
        double lambda = 0;
        for (unsigned int j=n-1; j>=i; j--)
        {
            curCost=0;
            unsigned int jp1 = j+1;
            unsigned int imnpj = imn+j;
            unsigned int imnpjp1 = imnpj + 1;  
            top_sz = jp1 - i;
            

            if (j<n-1 && size[imnpjp1]>1) 
            {
                // the default values
                size[imnpj] = size[imnpjp1]-1;
                
                left_i[j] = left_i[jp1];
                for (int tmp=0;tmp<d;tmp++) 
                {
                    cntr_i[j*d+tmp] = (cntr_i[jp1*d+tmp]*size[imnpjp1]-data[jp1*d+tmp])/size[imnpj];
                    left_i[j] -= ((data[jp1*d+tmp]-cntr_i[jp1*d+tmp])*(data[jp1*d+tmp]-cntr_i[j*d+tmp]));
                }
            }


            if (lambda==0 || theta==0 || sum_pow2(&data[jp1*d], d) >= lambda*theta) // data[jp1]*data[jp1] >= lambda*theta), i.e., no approximation
            {
                int sz = 1;
                cp_dbl(m, &data[j*d], d); // [j-sz+1];
                double SSE = 0;
                curCost = left[i-1][j-1];

                // update if necessary
                if (curCost < left_i[j])
                {
                    cp_dbl(&cntr_i[j*d], m, d);
                    left_i[j] = curCost;
                    size[imnpj] = sz;
                }

                double best_left = left_i[j];
                double *best_cntr = new double[d];
                cp_dbl(best_cntr, &cntr_i[j*d], d);
                int best_size = size[imnpj];

                cp_dbl(m_old, m, d);
                cp_dbl(sum, m_old, d);

                //int jmszp1 = j - sz + 1;
                
                double *cur_data = new double[d]();

                double *prev_left = &left[i-1][j-2];    // start of sz is 2

                const double *cur_data_ptr = &data[j*d];
                for (int sz=2; sz<=top_sz; sz++)
                {

                    // std::cout << i << j << sz << std::endl;

                    //jmszp1--; 
                    //cur_data = data[jmszp1];
                    cur_data_ptr -= d;
                    cp_dbl(cur_data, cur_data_ptr, d);

                    // m = (m_old*(sz-1)+data[jmszp1])/sz;
                    for (int tmp=0;tmp<d;tmp++)
                        sum[tmp] += cur_data[tmp];
                    
                    cp_dbl_div(m, sum, sz, d);   // m = sum/sz

                    for (int tmp=0;tmp<d;tmp++)
                        SSE += (cur_data[tmp]-m_old[tmp])*(cur_data[tmp]-m[tmp]);

                    //curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);
                    curCost = *prev_left-- + SSE;

                    // update if necessary
                    if (curCost < best_left) // left[i][j])
                    {
                        cp_dbl(best_cntr, m, d);
                        best_left = curCost;
                        best_size = sz;
                    }

                    cp_dbl(m_old, m, d);
                }
                cp_dbl(&cntr_i[j*d], best_cntr, d);
                left_i[j] = best_left;
                size[imnpj] = best_size;

                delete[] cur_data;
                cur_data = 0;
                delete[] best_cntr;
                best_cntr = 0;
            }
            else 
            {
                int stall = 0;
                int sz = 1;
                cp_dbl(m, &data[(j-sz+1)*d], d);
                double SSE = 0;
                curCost = left[i-1][j-1];

                if (curCost < left_i[j])
                {
                    cp_dbl(&cntr_i[j*d], m, d);
                    left_i[j] = curCost;
                    size[imnpj] = sz;
                    stall = 0;
                }
                // else 
                // {
                //     stall++;
                // }

                double best_left = left_i[j];
                double *best_cntr = new double[d];
                cp_dbl(best_cntr, &cntr_i[j*d], d);
                int best_size = size[imnpj];

                cp_dbl(m_old, m, d);
                cp_dbl(sum, m_old, d);
                //int jmszp1 = j - sz + 1;
                double *cur_data = new double[d];

                double *prev_left = &left[i-1][j-2];    // start of sz is 2
                const double *cur_data_ptr = &data[j*d];
                for (int sz=2; sz<=top_sz; sz++)
                {
                    //jmszp1--;   // j - sz
                    //m = (m_old*(sz-1)+data[jmszp1])/sz;
                    
                    //cur_data = data[jmszp1];
                    cur_data_ptr-=d;
                    cp_dbl(cur_data, cur_data_ptr, d);

                    add_dbl(sum, cur_data, d);
                    cp_dbl_div(m, sum, sz, d);
                    
                    for (int tmp=0;tmp<d;tmp++)
                        SSE += (cur_data[tmp]-m_old[tmp])*(cur_data[tmp]-m[tmp]);

                    //curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);
                    curCost = *prev_left-- + SSE;

                    // update if necessary
                    if (curCost < best_left) // left[i][j])
                    {
                        cp_dbl(best_cntr, m, d);
                        best_left = curCost;
                        best_size = sz;
                        stall = 0;
                    }
                    else if (sz>size[imnpjp1])    // we know j<n-1 since lambda!=0
                    {
                        stall++;

                        if (stall>=MAX_STALL)// && (SSE-left[i-1][j-sz])>=threshold) 
                        {
                            #ifdef VERBOSE
                                total_saved_operations+=top_sz-sz;
                            #endif
                            #ifdef VERBOSE2
                                std::cout << "i: " << i << ", j: " << j << ", data[jp1]**2: " << data[jp1]*data[jp1] << ", lambda: " << lambda << ", sz: " << sz << ", top_sz:" << top_sz << ", saved: " << top_sz-sz << std::endl;
                            #endif

                            break;
                        }
                    }
                    cp_dbl(m_old, m, d);
                }
                cp_dbl(&cntr_i[j*d], best_cntr, d);
                left_i[j] = best_left;
                size[imnpj] = best_size;

                delete[] cur_data;
                cur_data = 0;
                delete[] best_cntr;
                best_cntr=0;
            }

            // if approximation is requested and not in the last row
            if (theta>0 && i<k-1) 
            {
                if (size[imnpj]>1)
                {
                    double tmp = 0;
                    for (int tmp2=0; tmp2<d; tmp2++)
                        tmp+=pow2(data[j*d+tmp2]-cntr_i[j*d+tmp2]);
                    lambda =  tmp*size[imnpj]/(size[imnpj]-1);
                }
                else if (j>=2)
                {
                    lambda = left[i-1][j-1]-left[i-1][j-2];
                    //std::cout << "+";
                }
                else // may be imprved later
                    lambda = 0;
            } 
            else
            {
                lambda = 0;
            }

            // other entries in the last row are not required
            if (i==k-1 && theta>0)
                break;
        } // end of for j

        all_SSEs[i] = left_i[n-1];
        delete[] left[i-1];
        left[i-1] = 0;
        delete[] cntr[i-1];
        cntr[i-1] = 0;
        delete []cntr_i;
        cntr_i = 0;
    }  // end of for i


    //std::cout << "total updates: " << total_updates << std::endl;

    int i = k-1;
    int j = n-1;

    std::vector<int> sizes(k); 

    while(i>=0)
    {
        int cur_sz = (int)size[i*n+j];

        //sizes.insert(sizes.begin(), cur_sz);
        sizes[i] = cur_sz;

        i--;
        j-=cur_sz;
    }

    out_sizes = std::move(sizes);
    out_SSE = left[k-1][n-1];
    out_All_SSEs = std::move(all_SSEs);
    
    delete[] left[k-1];
    left[k-1] = 0;
    delete[] cntr[k-1];
    cntr[k-1] = 0;
    delete[] left;
    left = 0;
    delete[] cntr;
    cntr = 0;
    delete[] data;
    data = 0;

    // #ifdef VERBOSE
    if (verbose) 
    {
        std::cout << "100%" << std::endl;
    }
    // #endif
}


// for numpy arrays
// based on https://www.youtube.com/watch?v=3eCt2SUmJG0
// the improved version that does not return internals
void accelerated_sequence_clustering(int n, 
        double *data, 
        int k,
        // std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE
        )
{
    
    const int kn = k*n;

    std::vector<double> left(kn, DBL_MAX);
    std::vector<double> cntr(kn);
    std::vector<int>    size(kn);
    
    double m, m_old, SSE, sum = 0;

    left[0] = SSE = 0; 
    m = m_old = sum = cntr[0] = data[0]; 
    size[0] = 1;

    for (int j=1; j<n; j++)
    {
        sum += data[j];
        m = sum/(j+1);
        cntr[j] = m;
        
        SSE += (data[j] - m) * (data[j] - m_old);

        left[j] = SSE;
        size[j] = j+1;

        m_old = m;
    }

    for (int i=1;i<k; i++)
    {
        for (int j=n-1; j>=i; j--)
        {
            double curCost=0;
            int top_sz = j-i+1;
            int ixnpj = i*n+j;

            double sum;

            for (int sz=1; sz<=top_sz; sz++)
            {
                if (sz>1)
                {
                    sum+=data[j-sz+1];
                    m = sum/sz;
                    SSE += (data[j-sz+1]-m_old)*(data[j-sz+1]-m);

                    curCost = left[(i-1)*n+(j-sz)] + SSE; 
                }
                else  // sz=1
                {
                    sum = m = data[j];
                    SSE = 0;
                    curCost = left[(i-1)*n+(j-1)];
                }


                if (curCost < left[i*n+j])
                {
                    cntr[ixnpj] = m;
                    left[ixnpj] = curCost;
                    size[ixnpj] = sz;
                } 
                // else if (SSE-left[i-1][j-sz] >= left[i][j])
                // {
                //     break;
                // }
                                
                m_old = m;
            }
        }
    }

    int i = k-1;
    int j = n-1;

    std::vector<double> centers(k);
    std::vector<int> sizes(k); 

    while(i>0)
    {
        int cur_sz = (int)size[i*n+j];

        sizes[i] = cur_sz;
        
        //centers[i] = cntr[i*n+j];

        i--;
        j-=cur_sz;
    }

    out_sizes = std::move(sizes);
    out_SSE = left[kn-1];
}


// for numpy arrays
// based on https://www.youtube.com/watch?v=3eCt2SUmJG0
void basic_sequence_clustering(int n, 
        double *data, 
        int k,
        std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left
        )
{
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""
    //int n = sizeof(data)/sizeof(data[0]);
    //std::cout << n << std::endl;

    // for debug
    //std::cout << "The numpy version" << std::endl;
    
    std::vector<std::vector<double> > left(k+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<double> > cntr(k+1, std::vector<double>(n+1, 0));
    std::vector<std::vector<int> >    size(k+1, std::vector<int>(n+1, -1));
    

    double m, m_old, SSE;

    // precomputed values of the data points
    // std::vector<double> precomputed_m(n+1, 0);
    // std::vector<double> precomputed_SSE(n+1, 0);
    // double dSSE;

    for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            cntr[1][1] = m = data[0];

            //////////////////

            // precomputed_m[j] = data[n-1];
            // precomputed_SSE[j] = 0;
        }
        else
        {
            m = (m_old*(j-1)+data[j-1])/j;
            //m = compute_center(data, j);
            
            cntr[1][j] = m;
            
            SSE += (data[j-1] - m) * (data[j-1] - m_old);
            //left[1][j] = SSE = compute_SSE(data, j, m);
            left[1][j] = SSE;

            ///////////////////////
            // precomputed_m[j] = (precomputed_m[j-1]*(j-1)+data[n-j])/j;
            // dSSE = (data[n-j] - precomputed_m[j]) * (data[n-j] - precomputed_m[j-1]);
            // precomputed_SSE[j] = precomputed_SSE[j-1]+dSSE;
        }

        size[1][j] = j;
        m_old = m;
    }

    for (int i=2;i<k+1; i++)
    {
        for (int j=n; j>i-1; j--)
        {
            
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            int top_sz = j-i+1;

            // // imporves number of updates by initialization based on the results of the previous step
            // if (j<n && size[i][j+1]>1) 
            // {
            //     cntr[i][j] = (cntr[i][j+1]*size[i][j+1]-data[j])/(size[i][j+1]-1);
            //     left[i][j] = left[i][j+1] - ((data[j]-cntr[i][j+1])*(data[j]-cntr[i][j]));
            //     size[i][j] = size[i][j+1]-1;
            // }

            for (int sz=1; sz<=top_sz; sz++)
            {
                if (sz==1)
                {
                    m = data[j-sz];
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                    //m = compute_center(data, j-sz, j);
                    
                    m = (m_old*(sz-1)+data[j-sz])/sz;
                    SSE += (data[j-sz]-m_old)*(data[j-sz]-m);

                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);

                    //double tmp_m = precomputed_m[sz];
                    //double tmp_SSE = precomputed_SSE[sz];
                    // m = precomputed_m[sz];
                    // curCost = left[i-1][j-sz] + precomputed_SSE[sz];
                }

                if (curCost < left[i][j] || left[i][j]<0)
                {
                    cntr[i][j] = m;
                    left[i][j] = curCost;
                    size[i][j] = sz;
                } 
                // else if (SSE-left[i-1][j-sz] >= left[i][j])
                // {
                //     break;
                // }
                                
                m_old = m;
            }
        }
    }

    int i = k;
    int j = n;

    std::vector<double> centers;
    std::vector<int> sizes; 

    while(i>0)
    {
        int cur_sz = size[i][j];

        sizes.insert(sizes.begin(), cur_sz);
        centers.insert(centers.begin(), cntr[i][j]);

        i--;
        j-=cur_sz;
    }


    // for debug
    // int cur=0, en;
    // for(int ik=0; ik<k; ik++)
    // {
    //     std::cout << sizes[ik] << std::endl << "  ->  ";
    //     en = cur + sizes[ik];
    //     while (en - cur > 0)
    //     {
    //         std::cout << data[cur] << " ";
    //         cur++;
    //     }

    //     std::cout << ": " << centers[ik] << std::endl << std::endl;
    // }


    out_cntrs = centers;
    out_sizes = sizes;
    out_SSE = left[k][n];
    
    out_internal_size = size;
    out_internal_left = left;
}


// for a range of k values between k_min and k_max
// for numpy arrays
// based on https://www.youtube.com/watch?v=3eCt2SUmJG0
void basic_sequence_clustering(int n, 
        double *data, 
        int k_min,
        int k_max,
        std::vector<std::vector<double> > & out_cntrs2, 
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2,
        std::vector<std::vector<int> > & out_internal_size2,
        std::vector<std::vector<double> > & out_internal_left2
        )
{
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""

    // for debug
    //std::cout << "The numpy version" << std::endl;
    
    std::vector<std::vector<double> > left(k_max+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<double> > cntr(k_max+1, std::vector<double>(n+1, 0));
    std::vector<std::vector<int> >    size(k_max+1, std::vector<int>(n+1, -1));
    

    double m, m_old, SSE;
    for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            cntr[1][1] = m = data[0];
        }
        else
        {
            m = (m_old*(j-1)+data[j-1])/j;
            //m = compute_center(data, j);
            
            cntr[1][j] = m;
            
            SSE += (data[j-1] - m) * (data[j-1] - m_old);
            //left[1][j] = SSE = compute_SSE(data, j, m);
            left[1][j] = SSE;
        }

        size[1][j] = j;
        m_old = m;
    }

    for (int i=2;i<k_max+1; i++)
    {
        for (int j=n; j>i-1; j--)
        {
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            
            for (int sz=1; sz<j-i+2; sz++)
            {
                if (sz==1)
                {
                    m = data[j-sz];
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                    //m = compute_center(data, j-sz, j);
                    m = (m_old*(sz-1)+data[j-sz])/sz;
                    SSE += (data[j-sz]-m_old)*(data[j-sz]-m);
                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);
                }

                if (curCost < left[i][j] || left[i][j]<0)
                {
                    cntr[i][j] = m;
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                m_old = m;
            }
        }
    }

    std::vector<std::vector<double> > centers(k_max-k_min+1, std::vector<double>());
    std::vector<std::vector<int> > sizes(k_max-k_min+1, std::vector<int>()); 
    std::vector<double> SSEs;

    out_cntrs2.resize(k_max-k_min+1);
    out_sizes2.resize(k_max-k_min+1);
    out_SSE2.resize(k_max-k_min+1);

    out_internal_left2.resize(k_max+1);
    out_internal_size2.resize(k_max+1);

    for (int k=k_min; k<=k_max; k++)
    {
        int i = k;
        int j = n;

        while(i>0)
        {
            int cur_sz = size[i][j];

            sizes[k-k_min].insert(sizes[k-k_min].begin(), cur_sz);
            centers[k-k_min].insert(centers[k-k_min].begin(), cntr[i][j]);

            i--;
            j-=cur_sz;
        }


        // for debug
        // int cur=0, en;
        // for(int ik=0; ik<k; ik++)
        // {
        //     std::cout << sizes[ik] << std::endl << "  ->  ";
        //     en = cur + sizes[ik];
        //     while (en - cur > 0)
        //     {
        //         std::cout << data[cur] << " ";
        //         cur++;
        //     }

        //     std::cout << ": " << centers[ik] << std::endl << std::endl;
        // }


        out_cntrs2[k-k_min] = centers[k-k_min];
        out_sizes2[k-k_min] = sizes[k-k_min];
        out_SSE2[k-k_min] = left[k][n];

    }
    out_internal_left2 = left;
    out_internal_size2 = size;
}

///////// ************** for multivariate time-series **************///////////////

/*
void accelerated_sequence_clustering(int n,
        int ndims,
        double *data, 
        int k,
        std::vector<std::vector<double>> & out_cntrs_2d, 
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left
        )
{

    std::vector<std::vector<double> > left(k+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<std::vector<double> > > cntr(k+1, std::vector<std::vector<double> >(n+1, std::vector<double>(ndims, 0)));
    std::vector<std::vector<int> >    size(k+1, std::vector<int>(n+1, -1));
    

    double *m = new double[ndims];
    double *m_old = new double[ndims];
    double SSE = 0;

    for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            
            for (int d=0;d<ndims;d++) 
            {
                cntr[1][1][d] = m[d] = data[0*ndims+d];
            }            
            //////////////////

            // precomputed_m[j] = data[n-1];
            // precomputed_SSE[j] = 0;
        }
        else
        {
            for (int d=0;d<ndims;d++) 
            {
                m[d] = (m_old[d]*(j-1)+data[(j-1)*ndims+d])/j;
                cntr[1][j][d] = m[d];

                SSE += (data[(j-1)*ndims+d] - m[d]) * (data[(j-1)*ndims+d] - m_old[d]);
            }
            
            
            left[1][j] = SSE;

        }

        size[1][j] = j;
        
        for (int d=0;d<ndims;d++) 
        {
            m_old[d] = m[d];
        }
    }

    for (int i=2;i<k+1; i++)
    {
        for (int j=n; j>i-1; j--)
        {
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            
            for (int sz=1; sz<j-i+2; sz++)
            {
                if (sz==1)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = data[(j-1)*ndims+d];
                    }
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                    //m = compute_center(data, j-sz, j);
                    
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = (m_old[d]*(sz-1)+data[(j-sz)*ndims+d])/sz;
                        SSE += (data[(j-sz)*ndims+d]-m_old[d])*(data[(j-sz)*ndims+d]-m[d]);
                    }
                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);

                    //double tmp_m = precomputed_m[sz];
                    //double tmp_SSE = precomputed_SSE[sz];
                    // m = precomputed_m[sz];
                    // curCost = left[i-1][j-sz] + precomputed_SSE[sz];
                }

                if (curCost < left[i][j] || left[i][j]<0)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        cntr[i][j][d] = m[d];
                    }
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                for (int d=0;d<ndims;d++)
                {
                    m_old[d] = m[d];
                }
            }
        }
    }

    int i = k;
    int j = n;

    std::vector<std::vector<double> > centers; //(k, std::vector<double>(ndims, 0));
    std::vector<int> sizes; 

    while(i>0)
    {
        int cur_sz = size[i][j];

        sizes.insert(sizes.begin(), cur_sz);
        centers.insert(centers.begin(), cntr[i][j]);

        i--;
        j-=cur_sz;
    }

    out_cntrs_2d = centers;
    out_sizes = sizes;
    out_SSE = left[k][n];
    
    out_internal_size = size;
    out_internal_left = left;

    delete[] m;
    delete[] m_old;
}


// for a range of k values between k_min and k_max
// for numpy arrays
// based on https://www.youtube.com/watch?v=3eCt2SUmJG0
void accelerated_sequence_clustering(int n, 
        int ndims,
        double *data, 
        int k_min,
        int k_max,
        std::vector<std::vector<std::vector<double> > > & out_cntrs2_2d, 
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2,
        std::vector<std::vector<int> > & out_internal_size2,
        std::vector<std::vector<double> > & out_internal_left2
        )
{
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""

    std::vector<std::vector<double> > left(k_max+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<std::vector<double> > > cntr(k_max+1, std::vector<std::vector<double> >(n+1, std::vector<double>(ndims, 0)));
    std::vector<std::vector<int> >    size(k_max+1, std::vector<int>(n+1, -1));
    

    double *m = new double[ndims];
    double *m_old = new double[ndims];
    double SSE = 0;

        for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            
            for (int d=0;d<ndims;d++) 
            {
                cntr[1][1][d] = m[d] = data[0*ndims+d];
            }            
            //////////////////

            // precomputed_m[j] = data[n-1];
            // precomputed_SSE[j] = 0;
        }
        else
        {
            for (int d=0;d<ndims;d++) 
            {
                m[d] = (m_old[d]*(j-1)+data[(j-1)*ndims+d])/j;
                cntr[1][j][d] = m[d];

                SSE += (data[(j-1)*ndims+d] - m[d]) * (data[(j-1)*ndims+d] - m_old[d]);
            }
            
            
            left[1][j] = SSE;

        }

        size[1][j] = j;
        
        for (int d=0;d<ndims;d++) 
        {
            m_old[d] = m[d];
        }
    }


    for (int i=2;i<k_max+1; i++)
    {
        for (int j=n; j>i-1; j--)
        {
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            
            for (int sz=1; sz<j-i+2; sz++)
            {
                if (sz==1)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = data[(j-1)*ndims+d];
                    }
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                   
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = (m_old[d]*(sz-1)+data[(j-sz)*ndims+d])/sz;
                        SSE += (data[(j-sz)*ndims+d]-m_old[d])*(data[(j-sz)*ndims+d]-m[d]);
                    }
                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);

                }

                if (curCost < left[i][j] || left[i][j]<0)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        cntr[i][j][d] = m[d];
                    }
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                for (int d=0;d<ndims;d++)
                {
                    m_old[d] = m[d];
                }
            }
        }
    }

    std::vector<std::vector<std::vector<double> > > centers(k_max-k_min+1); //, std::vector<std::vector<double> >(ndims, 0));
    std::vector<std::vector<int> > sizes(k_max-k_min+1); 
    std::vector<double> SSEs;

    out_cntrs2_2d.resize(k_max-k_min+1);
    out_sizes2.resize(k_max-k_min+1);
    out_SSE2.resize(k_max-k_min+1);

    out_internal_left2.resize(k_max+1);
    out_internal_size2.resize(k_max+1);

    for (int k=k_min; k<=k_max; k++)
    {
        int i = k;
        int j = n;

        while(i>0)
        {
            int cur_sz = size[i][j];

            sizes[k-k_min].insert(sizes[k-k_min].begin(), cur_sz);
            centers[k-k_min].insert(centers[k-k_min].begin(), cntr[i][j]);

            i--;
            j-=cur_sz;
        }


        out_cntrs2_2d[k-k_min] = centers[k-k_min];
        out_sizes2[k-k_min] = sizes[k-k_min];
        out_SSE2[k-k_min] = left[k][n];

    }
    out_internal_left2 = left;
    out_internal_size2 = size;
}

*/


///////////////////////  for 2d numpy without output centers, due to swig error of vecvecvecdouble
///////// ************** for multivariate time-series **************///////////////

void basic_sequence_clustering_2d(int nrec,
        int ndims,
        double *data_2d, 
        int k,
        bool verbose,
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left
        )
{
    int n = nrec;
    //std::cout << n << " " << ndims << std::endl;

    std::vector<std::vector<double> > left(k+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<std::vector<double> > > cntr(k+1, std::vector<std::vector<double> >(n+1, std::vector<double>(ndims, 0)));
    std::vector<std::vector<int> >    size(k+1, std::vector<int>(n+1, -1));
    

    double *m = new double[ndims];
    double *m_old = new double[ndims];
    double SSE = 0;

    for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            
            for (int d=0;d<ndims;d++) 
            {
                cntr[1][1][d] = m[d] = data_2d[0*ndims+d];
            }            
            //////////////////

            // precomputed_m[j] = data[n-1];
            // precomputed_SSE[j] = 0;
        }
        else
        {
            for (int d=0;d<ndims;d++) 
            {
                m[d] = (m_old[d]*(j-1)+data_2d[(j-1)*ndims+d])/j;
                cntr[1][j][d] = m[d];

                SSE += (data_2d[(j-1)*ndims+d] - m[d]) * (data_2d[(j-1)*ndims+d] - m_old[d]);
            }
            
            left[1][j] = SSE;
        }

        size[1][j] = j;
        
        for (int d=0;d<ndims;d++) 
        {
            m_old[d] = m[d];
        }
    }

    for (int i=2;i<k+1; i++)
    {
        if (verbose) {
            std::cout << (double)i/(k+1)*100 << "%" << std::endl;
        }

        for (int j=n; j>i-1; j--)
        {
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            
            for (int sz=1; sz<j-i+2; sz++)
            {
                if (sz==1)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = data_2d[(j-1)*ndims+d];
                    }
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                    //m = compute_center(data, j-sz, j);
                    
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = (m_old[d]*(sz-1)+data_2d[(j-sz)*ndims+d])/sz;
                        SSE += (data_2d[(j-sz)*ndims+d]-m_old[d])*(data_2d[(j-sz)*ndims+d]-m[d]);
                    }
                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);

                    //double tmp_m = precomputed_m[sz];
                    //double tmp_SSE = precomputed_SSE[sz];
                    // m = precomputed_m[sz];
                    // curCost = left[i-1][j-sz] + precomputed_SSE[sz];
                }

                if (curCost < left[i][j] || left[i][j]<0) {
                    for (int d=0;d<ndims;d++) 
                    {
                        cntr[i][j][d] = m[d];
                    }
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                for (int d=0;d<ndims;d++) 
                {
                    m_old[d] = m[d];
                }
            }
        }
    }

    int i = k;
    int j = n;

    std::vector<std::vector<double> > centers; //(k, std::vector<double>(ndims, 0));
    std::vector<int> sizes; 

    while(i>0)
    {
        int cur_sz = size[i][j];

        sizes.insert(sizes.begin(), cur_sz);
        centers.insert(centers.begin(), cntr[i][j]);

        i--;
        j-=cur_sz;
    }

    //out_cntrs_2d = centers;
    out_sizes = sizes;
    out_SSE = left[k][n];
    
    out_internal_size = size;
    out_internal_left = left;

    delete[] m;
    delete[] m_old;

    if (verbose) {
        std::cout << "100%" << std::endl;
    }
}


// for a range of k values between k_min and k_max
// for numpy arrays
// based on https://www.youtube.com/watch?v=3eCt2SUmJG0
void basic_sequence_clustering_2d(int nrec, 
        int ndims,
        double *data_2d, 
        int k_min,
        int k_max,
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2,
        std::vector<std::vector<int> > & out_internal_size2,
        std::vector<std::vector<double> > & out_internal_left2
        )
{
    int n = nrec;
    //"""This function is based on paper "On the accelerated clustering of sequential data", SIAM 2002"""

    std::vector<std::vector<double> > left(k_max+1, std::vector<double>(n+1, -1));
    std::vector<std::vector<std::vector<double> > > cntr(k_max+1, std::vector<std::vector<double> >(n+1, std::vector<double>(ndims, 0)));
    std::vector<std::vector<int> >    size(k_max+1, std::vector<int>(n+1, -1));
    

    double *m = new double[ndims];
    double *m_old = new double[ndims];
    double SSE = 0;

        for (int j=1; j<n+1; j++)
    {
        if (j==1)
        {
            left[1][1] = SSE = 0;
            
            for (int d=0;d<ndims;d++) 
            {
                cntr[1][1][d] = m[d] = data_2d[0*ndims+d];
            }            
            //////////////////

            // precomputed_m[j] = data[n-1];
            // precomputed_SSE[j] = 0;
        }
        else
        {
            for (int d=0;d<ndims;d++) 
            {
                m[d] = (m_old[d]*(j-1)+data_2d[(j-1)*ndims+d])/j;
                cntr[1][j][d] = m[d];

                SSE += (data_2d[(j-1)*ndims+d] - m[d]) * (data_2d[(j-1)*ndims+d] - m_old[d]);
            }
            
            
            left[1][j] = SSE;

        }

        size[1][j] = j;
        
        for (int d=0;d<ndims;d++) 
        {
            m_old[d] = m[d];
        }
    }


    for (int i=2;i<k_max+1; i++)
    {
        for (int j=n; j>i-1; j--)
        {
            //left[i][j] = 1e12;  // np.inf for cost of partitioning j data point to i clusters
            double curCost=0;
            
            for (int sz=1; sz<j-i+2; sz++)
            {
                if (sz==1)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = data_2d[(j-1)*ndims+d];
                    }
                    SSE = 0;
                    curCost = left[i-1][j-sz];
                }
                else
                {
                   
                    for (int d=0;d<ndims;d++)
                    {
                        m[d] = (m_old[d]*(sz-1)+data_2d[(j-sz)*ndims+d])/sz;
                        SSE += (data_2d[(j-sz)*ndims+d]-m_old[d])*(data_2d[(j-sz)*ndims+d]-m[d]);
                    }
                    curCost = left[i-1][j-sz] + SSE; //compute_SSE(data, j-sz, j, m);

                }

                if (curCost < left[i][j] || left[i][j]<0)
                {
                    for (int d=0;d<ndims;d++)
                    {
                        cntr[i][j][d] = m[d];
                    }
                    left[i][j] = curCost;
                    size[i][j] = sz;
                }

                for (int d=0;d<ndims;d++)
                {
                    m_old[d] = m[d];
                }
            }
        }
    }

    std::vector<std::vector<std::vector<double> > > centers(k_max-k_min+1); //, std::vector<std::vector<double> >(ndims, 0));
    std::vector<std::vector<int> > sizes(k_max-k_min+1); 
    std::vector<double> SSEs;

    //out_cntrs2_2d.resize(k_max-k_min+1);
    out_sizes2.resize(k_max-k_min+1);
    out_SSE2.resize(k_max-k_min+1);

    out_internal_left2.resize(k_max+1);
    out_internal_size2.resize(k_max+1);

    for (int k=k_min; k<=k_max; k++)
    {
        int i = k;
        int j = n;

        while(i>0)
        {
            int cur_sz = size[i][j];

            sizes[k-k_min].insert(sizes[k-k_min].begin(), cur_sz);
            centers[k-k_min].insert(centers[k-k_min].begin(), cntr[i][j]);

            i--;
            j-=cur_sz;
        }


        //out_cntrs2_2d[k-k_min] = centers[k-k_min];
        out_sizes2[k-k_min] = sizes[k-k_min];
        out_SSE2[k-k_min] = left[k][n];

    }
    out_internal_left2 = left;
    out_internal_size2 = size;
}

// /* Need to use 1D index for accessing array elements */
// void modifyArray(int sizex, int sizey, double *arr) {
//   for (int i=0; i<sizex; i++) {
//   	for (int j=0; j<sizey; j++) {
//   		int n=i*sizey+j;

//   		arr[n] = i*j;
//   	}
//   }
// }