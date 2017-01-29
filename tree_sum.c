
/* File:    
 *  Algoritimo questao 1 - Soma em Arvore   
 *    tree_sum.c
 *
 * Purpose:    
 *    A tree sum program that uses MPI
 *
 * Compile:    
 *    mpicc -g -Wall -std=c99 -o tree_sum tree_sum.c
 * Usage:        
 *    mpiexec -n<number of processes> ./tree_sum
 *
 * Input:      
 *    One number for each process
 * Output:     
 *    A total sum, showed by process 0
 *
 * Algorithm:  
 *    Each process sends an element to another process in the way 
 *    that, in the end, process 0 prints the total sum.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  
#include <mpi.h>     /* For MPI functions, etc */
#include <math.h> 

void Read_data(int vec[], int vec2[], int my_rank, int comm_sz, MPI_Comm comm);
void Print_vector(int vec[], int comm_sz, char title[], int my_rank, MPI_Comm comm);
int soma_arvore(int vec[], int my_rank, int comm_sz, MPI_Comm comm); 

int main(void) {
   int comm_sz;               /* Number of processes    */
   int my_rank;               /* My process rank        */
   int *vec, *vec2;
   int reduce = 0;
   int total = 0;
   MPI_Comm comm;
   double start, finish, loc_elapsed, elapsed;

   /* Start up MPI */
   MPI_Init(NULL, NULL); 
   comm = MPI_COMM_WORLD;
   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
   
   vec = malloc(comm_sz*sizeof(int));
   vec2 = malloc(comm_sz*sizeof(int));

   /*Read input data*/
   Read_data(vec, vec2, my_rank, comm_sz, comm);

   /* Print input data */
   // if (my_rank == 0)
   //    printf("\n\n ===== input data =====\n");
   //    Print_vector(vec, comm_sz, "The vector is", my_rank, comm);
   
   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   total = soma_arvore(vec, my_rank, comm_sz, comm);
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0)
      printf("Elapsed time com MPI_Send e MPI_Recv: %e\n", elapsed);

   /*Print result*/ 
  //  if(my_rank == 0)
    // printf("\nSoma:  %d\n", total);

   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   MPI_Reduce(vec2, &reduce, 1, MPI_INT, MPI_SUM, 0, comm);
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0)
      printf("Elapsed time com  MPI_Reduce: %e\n", elapsed);
   /*Print result*/
  //  if(my_rank == 0)
    // printf("\nSoma: %d\n", reduce);

   free(vec);
   free(vec2);
   /* Shut down MPI */
   MPI_Finalize(); 

   return 0;
}  /* main */

/*-------------------------------------------------------------------*/
void Read_data(int vec[],int vec2[], int my_rank, int comm_sz, MPI_Comm comm) {
   int* a = NULL;
   int i;

   if (my_rank == 0){
      a = malloc(comm_sz * sizeof(int));
      for (i = 0; i < comm_sz; i++) a[i] = i + 1;//ele vai criar um vetor com o indece 
      MPI_Scatter(a, 1, MPI_INT, vec, 1, MPI_INT, 0, comm);
      MPI_Scatter(a, 1, MPI_INT, vec2, 1, MPI_INT, 0, comm);
      free(a);
   } else {
      MPI_Scatter(a, 1, MPI_INT, vec, 1, MPI_INT, 0, comm);
      MPI_Scatter(a, 1, MPI_INT, vec2, 1, MPI_INT, 0, comm);
   }
}  /* Read_data */
/*-------------------------------------------------------------------*/
void Print_vector(int vec[], int comm_sz, char title[], 
      int my_rank, MPI_Comm comm) {
   int* a = NULL;
   int i;
   
   if (my_rank == 0) {
      a = malloc(comm_sz * sizeof(int));
      MPI_Gather(vec, 1, MPI_INT, a, 1, MPI_INT, 0, comm);
      printf("%s\n", title);
      for (i = 0; i < comm_sz; i++) 
         printf("%.d ", a[i]);
      printf("\n");
      free(a);
   } else {
      MPI_Gather(vec, 1, MPI_INT, a, 1, MPI_INT, 0, comm);
   }

}  /* Print_vector */
/*-------------------------------------------------------------------*/
int soma_arvore(int vec[], int my_rank, int comm_sz, MPI_Comm comm) {
    
    int soma_local = *vec;
    int temp;
    int parceiro;
    unsigned bitmask = (unsigned) 1;

    while (bitmask < comm_sz) {
        parceiro = my_rank ^ bitmask;

        if (my_rank < parceiro) {
            if (parceiro < comm_sz) {
                MPI_Recv(&temp, 1, MPI_INT, parceiro, 0, comm, 
                      MPI_STATUS_IGNORE);
                soma_local += temp;
            }
            bitmask <<= 1;
        } else {
                MPI_Send(&soma_local, 1, MPI_INT, parceiro, 0, comm); 
            break;
        }
    }
    return soma_local;
}
/*-------------------------------------------------------------------*/

