#include <vector>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "accelerated_sequence_clustering.h"



using namespace std;
using namespace std::chrono;

int main()
{
    int n = 512;
    int d = 512;
    int k = 16;

    double *b = new double[n*d];
    for (int i=0;i<n*d;i++)
        b[i] = cos(i)/(0.001+sin(i*i));

    vector<int> sizes;

    double real_SSE, approximated_SSE;

    auto start1 = high_resolution_clock::now();
    // no approximation basic=0, 
    accelerated_sequence_clustering_approximated3_2d(n, d, b, k, 1, 0, sizes, real_SSE);   // theta = 0 means no approaximation
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);

    auto start2 = high_resolution_clock::now();
    // approximation is applied
    accelerated_sequence_clustering_approximated3_2d(n, d, b, k, 1, 15, sizes, approximated_SSE);   // theta = 0 means no approaximation
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);


    std::cout << std::fixed;
    std::cout << "Real SSE: " << real_SSE << 
                 ", Approximated SSE: " << approximated_SSE << 
                 ", SSE Difference: " << approximated_SSE - real_SSE << 
                 ", Gap%: " <<  100*(approximated_SSE - real_SSE)/real_SSE << 
                 std::endl;

    std::cout << "Basic Algorithm Time: " << duration1.count() << " usec, Accelerated Algorithm Time: " << duration2.count() << " usec, Speedup: " << double(duration1.count())/duration2.count() << std::endl;


    delete[] b;
    b = 0;

    return 0;
}