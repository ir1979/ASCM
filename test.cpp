#include <vector>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include "accelerated_sequence_clustering.h"

using namespace std;
using namespace std::chrono;

// A function that generates random numbers using mersenne twister




/**
 * Generates a random vector of doubles.
 *
 * @param n the size of the vector to generate
 * @param seed the seed value for random number generation (default: 0)
 * @param max_val the maximum value for the generated random numbers (default: 10)
 *
 * @return a vector of doubles containing randomly generated values
 *
 * @throws None
 */
std::vector<double> generate_random_vector(int n, unsigned long seed=0, int max_val=10)
{
    // First create an instance of an engine.
    // random_device rnd_device;
    // Specify the engine and distribution.
    // mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    mt19937 mersenne_engine {seed};  // Generates random integers initialized by the seed
    
    uniform_int_distribution<int> dist {1, max_val};

    auto gen = [&dist, &mersenne_engine](){
                return dist(mersenne_engine);
            };

    std::srand(seed);
    std::vector<double> v(n);
    std::generate(v.begin(), v.end(), gen);
    return v;
}

int main()
{
    int n = 10000;    // number of records
    int d = 2;      // number of dimensions
    int k = 16;     // number of clusters
    int seed = 0;   // seed used for random number generation

    cout << "Processing a synthetic dataset (initizlied with seed=" 
         << seed << ") containing " 
         << n 
         << " records and " 
         << d 
         << " dimensions..." << endl;

    cout << "The dataset is to be clustered into " 
         << k << " clusters." << endl;

    int max_stall = 10; // maximum number of stalls after which to stop the search within the algorithm
    double theta = 1.5;

    vector<double> a = generate_random_vector(n*d, seed);

    double *b = &a[0];  // copy of a in b

    vector<int> sizes;

    // these matrices are not used by now
    vector<vector<int> >    internal_size_matrix;
    vector<vector<double> > internal_left_matrix;

    double real_SSE, approximated_SSE;

    auto start1 = high_resolution_clock::now();

    // no approximation basic: theta=0, 
    //accelerated_sequence_clustering_approximated3_2d(n, d, b, k, 0, 0, sizes, real_SSE);   // theta = 0 means no approaximation
    basic_sequence_clustering_2d(n, d, b, k, sizes, real_SSE, internal_size_matrix, internal_left_matrix);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);

    auto start2 = high_resolution_clock::now();
    // approximation is applied, theta=1.5 
    accelerated_sequence_clustering_approximated3_2d(n, d, b, k, max_stall, theta, sizes, approximated_SSE);   // theta = 0 means no approaximation
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);


    std::cout << std::fixed;
    std::cout << "Real SSE: " << real_SSE << 
                 ", Approximated SSE: " << approximated_SSE << 
                 ", SSE Difference: " << approximated_SSE - real_SSE << 
                 ", Gap%: " <<  100*(approximated_SSE - real_SSE)/real_SSE << 
                 std::endl;

    std::cout << "Basic Algorithm Time: " << duration1.count() << " msec, Accelerated Algorithm Time: " << duration2.count() << " msec, Speedup: " << double(duration1.count())/duration2.count() << std::endl;

    return 0;
}