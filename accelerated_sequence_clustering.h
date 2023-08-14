#include <vector>


extern void accelerated_sequence_clustering_approximated(int n, 
        double *data, 
        int k,
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE
        , std::vector<std::vector<int> > & out_internal_size
        , std::vector<std::vector<double> > & out_internal_left);
        

extern void accelerated_sequence_clustering_approximated(std::vector<double> data, 
        int k, 
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE
        , std::vector<std::vector<int> > & out_internal_size
        , std::vector<std::vector<double> > & out_internal_left);

extern void accelerated_sequence_clustering_approximated2(int n, 
        double *data, 
        int k,
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<unsigned int> & out_sizes,
        double & out_SSE
        , std::vector<unsigned int> & out_internal_size_flat
        , std::vector<double> & out_internal_left_flat);


extern void accelerated_sequence_clustering_approximated2(std::vector<double> data, 
        int k, 
        int MAX_STALL,
        double theta,
        std::vector<double> & out_cntrs, 
        std::vector<unsigned int> & out_sizes_unsigned,
        double & out_SSE
        , std::vector<unsigned int> & out_internal_size_flat
        , std::vector<double> & out_internal_left_flat);

void accelerated_sequence_clustering_approximated3(std::vector<double> data, 
        int kk, 
        int MAX_STALL,
        double theta,
        std::vector<int>& out_sizes3,
        double & out_SSE3);

void accelerated_sequence_clustering_approximated3(int nn, 
        double *data, 
        int kk,
        int MAX_STALL,
        double theta,
        std::vector<int> & out_sizes3,
        double & out_SSE3);

extern void accelerated_sequence_clustering(int n, 
        double *data, 
        int k,
        std::vector<int> & out_sizes,
        double & out_SSE);
        

extern void accelerated_sequence_clustering(std::vector<double> data, 
        int k, 
        std::vector<int>& out_sizes,
        double & out_SSE);


extern void basic_sequence_clustering(int n, 
        double *data, 
        int k,
        std::vector<double> & out_cntrs, 
        std::vector<int> & out_sizes,
        double & out_SSE
        , std::vector<std::vector<int> > & out_internal_size
        , std::vector<std::vector<double> > & out_internal_left);
        

extern void basic_sequence_clustering(std::vector<double> data, 
        int k, 
        std::vector<double>& out_cntrs, 
        std::vector<int>& out_sizes,
        double & out_SSE
        , std::vector<std::vector<int> > & out_internal_size
        , std::vector<std::vector<double> > & out_internal_left);



extern void basic_sequence_clustering(int n, 
        double *data, 
        int k_min,
        int k_max,
        std::vector<std::vector<double> > & out_cntrs2, 
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2
        , std::vector<std::vector<int> > & out_internal_size2
        , std::vector<std::vector<double> > & out_internal_left2);

extern void basic_sequence_clustering(std::vector<double> data, 
        int k_min,
        int k_max, 
        std::vector<std::vector<double> > & out_cntrs2, 
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2
        , std::vector<std::vector<int> > & out_internal_size2
        , std::vector<std::vector<double> > & out_internal_left2);

extern void basic_sequence_clustering_2d(int nrec,
        int ndims,
        double *data_2d, 
        int k,
        bool verbose,
        std::vector<int> & out_sizes,
        double & out_SSE,
        std::vector<std::vector<int> > & out_internal_size,
        std::vector<std::vector<double> > & out_internal_left
        );

extern void basic_sequence_clustering_2d(int nrec, 
        int ndims,
        double *data_2d, 
        int k_min,
        int k_max,
        std::vector<std::vector<int> > & out_sizes2,
        std::vector<double> & out_SSE2,
        std::vector<std::vector<int> > & out_internal_size2,
        std::vector<std::vector<double> > & out_internal_left2
        );

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
        );
        

// extern void modifyArray(int sizex, int sizey, double *arr);