/* example.i */
%module accelerated_sequence_clustering
%{

#define SWIG_FILE_WITH_INIT
/* Put header files here or function declarations like below */
#include <iostream>
#include <vector>

// extern void accelerated_sequence_clustering(std::vector<double> data, int k, std::vector<double> & out_cntr, std::vector<int> & out_sizes, double & out_SSE);  // for lists
// extern void accelerated_sequence_clustering(int n, double *data, int k, std::vector<double> & out_cntrs2, std::vector<int> & out_sizes2, double & out_SSE2);   // for vectors

// extern void accelerated_sequence_clustering(int n, double *data, int k_min, int k_max, std::vector<std::vector<double> > & out_cntrs, std::vector<std::vector<int> > & out_sizes, std::vector<double> & out_SSE);
// extern void accelerated_sequence_clustering(std::vector<double> data, int k_min, int k_max, std::vector<std::vector<double> > & out_cntrs, std::vector<std::vector<int> > & out_sizes, std::vector<double> & out_SSE);

#include "accelerated_sequence_clustering.h"

using namespace std;
%}

// %include "windows.i"

%include "numpy.i"
%init %{
  import_array();
%}

%include "std_vector.i"
namespace std 
{
  %template(VecDouble) vector<double>;
  %template(VecInt) vector<int>;
  %template(VecUInt) vector<unsigned>;
  
  %template(VecVecDouble) vector<vector<double> >;
  %template(VecVecInt) vector<vector<int> >;

}

// %{
// #   define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS 
// %}


%apply (int DIM1, double *IN_ARRAY1) {(int n, double *data)};
%apply (int DIM1, int DIM2, double *IN_ARRAY2) {(int nrec, int ndims, double *data_2d)};
%apply (int DIM1, int DIM2, double *IN_ARRAY2) {(int nrows, int ncols, double *seq_data)};
// %apply (int DIM1, int DIM2, double *IN_ARRAY2, int, std::vector<int> & OUTPUT, 
//        double & OUTPUT, std::vector<std::vector<int> > & OUTPUT, std::vector<std::vector<double> > & OUTPUT) 
//        {(int nrec, int ndims, double *data_2d, int k, std::vector<int> & out_sizes, double & out_SSE, 
//         std::vector<std::vector<int> > & out_internal_size, std::vector<std::vector<double> > & out_internal_left)};
%apply int {int k};
%apply int {int kk};
%apply int {int nn};
%apply bool {bool verbose};

// %apply unsigned int {int uk};
// %apply unsigned int {int un};


%apply int {int MAX_STALL};
%apply double {double theta};
%apply int {int k_min};
%apply int {int k_max};
%apply std::vector<double> & OUTPUT { std::vector<double> & out_cntrs };


%apply std::vector<int> & OUTPUT { std::vector<int> & out_sizes3 };
%apply double & OUTPUT { double &out_SSE3 };
%apply long int & OUTPUT { long int &total_saved_operations };

%apply std::vector<int> & OUTPUT { std::vector<int> & out_sizes };
%apply std::vector<double> & OUTPUT { std::vector<double> & out_All_SSEs };
%apply double & OUTPUT { double &out_SSE };
%apply std::vector<std::vector<int> > & OUTPUT { std::vector<std::vector<int> >& out_internal_size};
%apply std::vector<std::vector<double> > & OUTPUT { std::vector<std::vector<double> >& out_internal_left};

// %apply std::vector<unsigned int> & OUTPUT { std::vector<unsigned int> & out_sizes_unsigned };
// %apply std::vector<unsigned int> & OUTPUT { std::vector<unsigned int> & out_internal_size_flat};
// %apply std::vector<double> & OUTPUT { std::vector<double>& out_internal_left_flat};


%apply std::vector<std::vector<double> > & OUTPUT { std::vector<std::vector<double> >& out_cntrs2 };
%apply std::vector<std::vector<int> > & OUTPUT { std::vector<std::vector<int> >& out_sizes2 };
%apply std::vector<double> & OUTPUT { std::vector<double> &out_SSE2 };
%apply std::vector<std::vector<int> > & OUTPUT { std::vector<std::vector<int> >& out_internal_size2 };
%apply std::vector<std::vector<double> > & OUTPUT { std::vector<std::vector<double> >& out_internal_left2};

// %apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int sizex, int sizey, double *arr)};

%include "accelerated_sequence_clustering.h"

