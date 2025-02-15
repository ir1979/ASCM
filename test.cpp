#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include "accelerated_sequence_clustering.h"

using namespace std;
using namespace std::chrono;

const bool STORE_DATASET = false;           // whether to store the generated dataset
const bool RUN_ALGORITHM = true;      // whether to only store the generated dataset and not apply algorithm to it

void save_csv(string filename, std::vector<double> data, int n, int d)
{
    ofstream out;
    out.open(filename, ios::out);
    for (int i=0; i<data.size(); i++)
        if (i%d==d-1)
            out << data[i] << endl;
        else
            out << data[i] << ", ";
    out.close();
}


/**
 * A function that generates random numbers using mersenne twister
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

    int max_stall = 10; // maximum number of stalls after which to stop the search within the algorithm
    double theta = 25;

    int* ns = new int[4]{5000, 10000, 50000, 100000};
    for (int i=0;i<4;i++){
        n = ns[i];
        for (seed=0;seed<1;seed++){
                    
            cout << "Processing a synthetic dataset (initizlied with seed=" 
                << seed << ") containing " 
                << n 
                << " records and " 
                << d 
                << " dimensions..." << endl;

            cout << "The dataset is to be clustered into " 
                << k << " clusters." << endl;

            vector<double> a = generate_random_vector(n*d, seed);

            if (STORE_DATASET) {
                string filename = "datasets/Syn_n_" + to_string(n) + "_d_" + to_string(d) + "_seed_" + to_string(seed) + ".csv";
                save_csv(filename, a, n, d);
                cout << filename << " is saved." << endl;
            }

            // to save csv file only once
            if (!RUN_ALGORITHM)
                continue;

            double *b = &a[0];  // copy of a in b

            vector<int> sizes;

            // these matrices are not used by now
            vector<vector<int> >    internal_size_matrix;
            vector<vector<double> > internal_left_matrix;

            double real_SSE, approximated_SSE;

            auto start1 = high_resolution_clock::now();

            // no approximation basic: theta=0, 
            //accelerated_sequence_clustering_approximated3_2d(n, d, b, k, 0, 0, sizes, real_SSE);   // theta = 0 means no approaximation
            basic_sequence_clustering_2d(n, d, b, k, false, sizes, real_SSE, internal_size_matrix, internal_left_matrix);
            auto stop1 = high_resolution_clock::now();
            auto duration1 = duration_cast<milliseconds>(stop1 - start1);

            vector<double> all_SSEs;
            long int total_saved_operations = 0;
            auto start2 = high_resolution_clock::now();
            // approximation is applied, theta=1.5 
            accelerated_sequence_clustering_approximated3_2d(n, d, b, k, max_stall, theta, false, sizes, approximated_SSE, all_SSEs, total_saved_operations);   // theta = 0 means no approaximation
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<milliseconds>(stop2 - start2);

            std::cout << std::fixed;
            std::cout << "Real SSE: " << real_SSE << 
                        ", Approximated SSE: " << approximated_SSE << 
                        ", SSE Difference: " << approximated_SSE - real_SSE << 
                        ", Gap: " <<  100*(approximated_SSE - real_SSE)/real_SSE << "%" << 
                        std::endl;

            std::cout << "Basic Algorithm Time: " << duration1.count() << " msec, Accelerated Algorithm Time: " << duration2.count() << " msec, Speedup: " << double(duration1.count())/duration2.count() << std::endl << std::endl;
        }
    }


    return 0;
}