/* File:      
 *    Algoritimo Questao 2 
 *    prefix_sum.c
 *
 * Purpose:    
 *    A prefix sum program that uses MPI
 *
 * Compile:    
 *    mpicc -g -Wall -std=c99 -o prefix_sum prefix_sum.c
 * Usage:        
 *    mpiexec -n<number of processes> ./prefix_sum
 *
 * Input:      
 *    None, it is the rank of each process
 * Output:     
 *    Value from the comm_sz-1 with the total sum of all the ranks
 *
 * Algorithm:  
 *    Each process sends his rank 
 *    to the next in a way that the 
 *    last process has the total sum
 *
 */

#include <stdio.h>
#include <stdlib.h>  
#include <string.h> 
#include <mpi.h>     /* For MPI functions, etc */
#include <math.h>

int Log2(int n); 
void soma_prefixada(int *sum, int my_rank, int comm_sz, MPI_Comm comm);

int main(void) {
   int comm_sz;               /* Number of processes    */
   int my_rank;               /* My process rank        */
   MPI_Comm comm;
   int sum;
   int sum_scan, result_scan;
   double start, finish, loc_elapsed, elapsed;

   /* Start up MPI */
   MPI_Init(NULL, NULL); 
   comm = MPI_COMM_WORLD;
   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

   sum = my_rank;
   sum_scan = my_rank;
 
   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   soma_prefixada(&sum, my_rank, comm_sz, comm);
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm_sz-1, comm);

   /*Print result*/
   if (my_rank == comm_sz-1){
        printf("Elapsed time com MPI_Send e MPI_Recv: %e\n", elapsed); 
   }
   /*Start the other funtion that uses Scan*/
   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   MPI_Scan(&sum_scan, &result_scan, 1, MPI_INT, MPI_SUM, comm);
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm_sz-1, comm);

   /*Print result*/
   if (my_rank == comm_sz-1){
      printf("Elapsed time com MPI_Scan: %e\n", elapsed); 
   }
   
   /* Shut down MPI */
   MPI_Finalize(); 

   return 0;
}  /* main */

/*-------------------------------------------------------------------*/
void soma_prefixada(int *sum, int my_rank,int comm_sz, MPI_Comm comm) {
   
   int soma_local = *sum;
   int altura = 0, divisor = 2, core_difference = 1;

   while(altura < log2(comm_sz)+1){
         if(my_rank < comm_sz-core_difference){
            MPI_Send(&soma_local, 1, MPI_INT, my_rank+core_difference, 0,
            MPI_COMM_WORLD);
      }  
         if(my_rank>=core_difference){
            int recebido;
            MPI_Recv(&recebido, 1, MPI_INT, my_rank-core_difference,
            0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            soma_local += recebido;
         } 
         core_difference *= 2;
         divisor *= 2;
         altura++;
   }
   *sum = soma_local;
}  
