/*
 * This is a mpi version of bellman_ford algorithm
 * Compile: mpic++ -std=c++11 -o mpi_bellman_ford mpi_bellman_ford.cpp
 * Run: mpiexec -n <number of processes> ./mpi_bellman_ford <input file>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "mpi.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
    int N; //number of vertices
    int *mat; // the adjacency matrix

    void abort_with_error_message(string msg) {
        std::cerr << msg << endl;
        abort();
    }

    //translate 2-dimension coordinate to 1-dimension
    int convert_dimension_2D_1D(int x, int y, int n) {
        return x * n + y;
    }

    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        if (!inputf.good()) {
            abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
        }
        inputf >> N;
        //input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
        assert(N < (1024 * 1024 * 20));
        mat = (int *) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                inputf >> mat[convert_dimension_2D_1D(i, j, N)];
            }
        return 0;
    }

    int print_result(bool has_negative_cycle, int *dist) {
        std::ofstream outputf("output.txt", std::ofstream::out);
        if (!has_negative_cycle) {
            for (int i = 0; i < N; i++) {
                if (dist[i] > INF)
                    dist[i] = INF;
                outputf << dist[i] << '\n';
            }
            outputf.flush();
        } else {
            outputf << "FOUND NEGATIVE CYCLE!" << endl;
        }
        outputf.close();
        return 0;
    }

    int print_result1(bool has_negative_cycle, int *dist) {
        std::ofstream outputf("output.txt", std::ofstream::out);
        if (!has_negative_cycle) {
            for (int i = 0; i < N; i++) {
                if (dist[i] > INF)
                    dist[i] = INF;
                outputf << dist[i] << '\n';
            }
            outputf.flush();
        } else {
            outputf << "FOUND NEGATIVE CYCLE!" << endl;
        }
        outputf.close();
        return 0;
    }
}//namespace utils

// you may add some helper functions here.

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param my_rank the rank of current process
 * @param p number of processes
 * @param comm the MPI communicator
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int *mat, int *dist, bool *has_negative_cycle) 
{
       
    //step 1: broadcast N
    	MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    	MPI_Barrier(MPI_COMM_WORLD);
    	int v = n;
    	int e = n*n;
		int src = 0;
		int a,b,weight;
	//	printf("process %d step %d n = %d\n",my_rank,1,n);

    //step 2: find local task range 
		int ll = n/p*my_rank*n,rr = n/p*(my_rank+1)*n;	
		if (my_rank == (p-1) && rr != (e)) rr = e;
		int i,j;
	//	printf("process %d step %d ll-rr:%d-%d\n",my_rank,2,ll,rr);

    //step 3: allocate and initiate local memory
		if (my_rank) {dist = (int*)malloc(n*sizeof(int));mat = (int*)malloc(n*n*sizeof(int));has_negative_cycle=(bool*)malloc(sizeof(bool));}
		for(i = 0;i < v;++i)  {dist[i] = INF;}
		int *g_dist = (int*)malloc(n*sizeof(int));
		for(i = 0;i < v;++i)  {g_dist[i] = INF;}
    	dist[src] = 0;g_dist[src] = 0;
    	MPI_Barrier(comm);
	//	printf("process %d step %d\n",my_rank,3);

    //step 4: broadcast matrix mat
		if ( my_rank == 0)
		{
			for (int pp = 1;pp < p; pp++)
			{
				int lll = n/p*pp*n,rrr = n/p*(pp+1)*n;	
				if (pp == (p-1) && rrr != (e)) rrr = e;
				MPI_Send(mat+lll,rrr-lll,MPI_INT,pp,0,comm);
		//		printf("process 0 send ll-rr:%d-%d\n",lll,rrr);
			}		
		}
		else
		{
			MPI_Recv(mat+ll,rr-ll,MPI_INT,0,0,comm,MPI_STATUS_IGNORE);
	//		printf("process %d recv ll-rr:%d-%d\n",my_rank,ll,rr);
		}
		MPI_Barrier(comm);
	//	printf("process %d step %d\n",my_rank,4);

    //step 5: bellman-ford algorithm
		// v-1 ierteration
		int done = 0, all_done = 0, t1, t2, now;
	//	printf("process %d begin to work\n",my_rank);
		int iter_n = v+1;
		int a_l = ll/n, a_r = rr/n;
	   	for(i = 1;i <= iter_n;++i)
		{
		    	// relaxation
			done = 1;
			for (j=ll, a=a_l; a<a_r; a++)	
				for (b=0; b<n; b++)
				{
		        		weight = mat[j];
						t1 = dist[a]+weight; 
		        		if(t1 < g_dist[b])
						{	
							g_dist[b] = t1;
							if (t1<dist[b])dist[b] = t1;
							done = 0;
		       			}	
						j++;
				}
			all_done = 1;
			MPI_Barrier(comm);	
			MPI_Allreduce(&done, &all_done, 1, MPI_INT, MPI_LAND, comm);
			MPI_Allreduce(g_dist, dist, n, MPI_INT, MPI_MIN, comm);
			MPI_Barrier(comm);	
			if (all_done || dist[0]<0) {break;}
		}

    // negative cycle detection
		if (my_rank){free(dist);free(mat);free(has_negative_cycle);}
    	else *has_negative_cycle=!all_done||(dist[0]<0);


}

int main(int argc, char **argv) {
    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    string filename = argv[1];

    int *dist;
    bool has_negative_cycle = false;

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p;//number of processors
    int my_rank;//my global rank
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    //only rank 0 process do the I/O
    if (my_rank == 0) {
        assert(utils::read_file(filename) == 0);
        dist = (int *) malloc(sizeof(int) * utils::N);
    }

    //time counter
    double t1, t2;
    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    //bellman-ford algorithm
    bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle);
    MPI_Barrier(comm);

    //end timer
    t2 = MPI_Wtime();

    if (my_rank == 0) {
        std::cerr.setf(std::ios::fixed);
        std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
        utils::print_result1(has_negative_cycle, dist);
        free(dist);
        free(utils::mat);
    }
    MPI_Finalize();
    return 0;
}
