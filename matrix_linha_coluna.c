/*
 * Algoritimo Questao 3
 * File:     matrix_linha_coluna.c
 *
 * Purpose:  Implement matrix vector multiplication when the matrix
 *           has a block column distribution
 *
 * Compile:  mpicc -g -Wall -o matrix_linha_coluna matrix_linha_coluna.c
 * Run:      mpiexec -n <comm_sz> ./matrix_linha_coluna
 *
 * Input:    order of matrix, matrix, vector
 * Output:   product of matrix and vector.
 *
 * Notes:    The matrix should be square and its order should be 
 *           evenly divisible by comm_sz
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  
#include <mpi.h>     /* For MPI functions, etc */
#include <math.h> 
#include <time.h>

void Get_dims(int* m_p, int* local_m_p, int* n_p, int* local_n_p,
      int my_rank, int comm_sz, MPI_Comm comm);
void Build_derived_type(int m, int local_m, int n, int local_n,
      MPI_Datatype* col);
void Read_matrix(double local_A[], double local_D[], int local_m, int m, 
      int local_n, int n, MPI_Datatype col, int my_rank, MPI_Comm comm);
void Print_matrix(double local_A[], int m, int local_n, 
      int n, MPI_Datatype col, int my_rank, MPI_Comm comm);
void Read_vector(double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm);
void Print_vector(double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm);
void Mat_vect_mult_colunas(double local_A[], double local_B[], double local_C[], 
      int local_m, int m, int n, int local_n, int comm_sz, MPI_Comm comm,
      int my_rank);
void Mat_vect_mult_row(double local_D[], double local_B[], 
      double local_E[], int local_m, int n, int local_n, 
      MPI_Comm comm);

int main(void) {
   int comm_sz;               /* Number of processes    */
   int my_rank;               /* My process rank        */
   int m, n;
   int local_m, local_n;
   double* local_A;
   double* local_B;
   double* local_C;
   double* local_D;
   double* local_E;
   
   MPI_Comm comm;
   MPI_Datatype col;
   double start, finish, loc_elapsed, elapsed;

   /* Start up MPI */
   MPI_Init(NULL, NULL); 
   comm = MPI_COMM_WORLD;
   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
   
   Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);

   local_A = malloc(m*local_n*sizeof(double));
   local_B = malloc(local_n*sizeof(double));
   local_C = malloc(local_m*sizeof(double));
   local_D = malloc(n*local_m*sizeof(double));
   local_E = malloc(local_m*sizeof(double));

   Build_derived_type(m, local_m, n, local_n, &col);
   
   Read_matrix(local_A, local_D, local_m, m, local_n, n, col, my_rank, comm);

   //Print_matrix(local_A, m, local_n, n, col, my_rank, comm);

   Read_vector(local_B, n, local_n, my_rank, comm);

   // if (my_rank == 0)
   //       printf("\nThe vector is:\n"); 
   // Print_vector(local_B, n, local_m, my_rank, comm);

   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   /*Function with columns*/
   Mat_vect_mult_colunas(local_A, local_B, local_C, local_m, m, n, 
            local_n, comm_sz, comm, my_rank);   
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0)
         printf("Elapsed time com colunas = %e", elapsed);

   //Print_vector(local_C, m, local_m, my_rank, comm);
   
   MPI_Barrier(comm);
   start = MPI_Wtime();
   
   /*Function with rows*/
   Mat_vect_mult_row(local_D, local_B, local_E, local_m, n, local_n, comm);
   
   finish = MPI_Wtime();
   loc_elapsed = finish-start;
   MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   if (my_rank == 0)
         printf("\nElapsed time com linhas = %e\n", elapsed);

   //Print_vector(local_E, m, local_m, my_rank, comm);

   free(local_A);
   free(local_B);
   free(local_C);
   free(local_D);
   free(local_E);
   MPI_Type_free(&col);
   /* Shut down MPI */
   MPI_Finalize(); 

   return 0;
}  /* main */
/*-------------------------------------------------------------------*/
void Get_dims(int* m_p, int* local_m_p, int* n_p,int* local_n_p,
      int my_rank, int comm_sz, MPI_Comm comm) {
 
   if (my_rank == 0) {
      // printf("Enter the order of the matrix\n");
      // scanf("%d", m_p);
      *m_p = 1024;//definicao manual do tamanho da matriz
   }
   MPI_Bcast(m_p, 1, MPI_INT, 0, comm);
   *n_p = *m_p;
   
   *local_m_p = *m_p/comm_sz;
   *local_n_p = *n_p/comm_sz;
}  /* Get_dims */
/*-------------------------------------------------------------------*/
void Build_derived_type(int m, int local_m, int n, int local_n,
      MPI_Datatype* col) {
   MPI_Datatype vect;

   MPI_Type_vector(m /*count*/, local_n /*blocklength*/, n /*stride*/, 
      MPI_DOUBLE, &vect);

   /* Resize the new type so that it has the extent of local_n doubles */
   MPI_Type_create_resized(vect, 0, local_n*sizeof(double), col);
   MPI_Type_commit(col);
}  /* Build_derived_type */
/*-------------------------------------------------------------------*/
void Read_matrix(double local_A[], double local_D[], int local_m, int m, 
      int local_n, int n, MPI_Datatype col, int my_rank, MPI_Comm comm) {
   double* A = NULL;
   int i, j;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));
      for (i = 0; i < m; i++)
         for (j = 0; j < n; j++)
            A[i*n+j] = i+j+1;

      MPI_Scatter(A, 1, col, local_A, m*local_n, MPI_DOUBLE, 0, comm);
      MPI_Scatter(A, local_m*n, MPI_DOUBLE, local_D, local_m*n, MPI_DOUBLE, 0, comm);
      free(A);
   } else {
      MPI_Scatter(A, 1, col, local_A, m*local_n, MPI_DOUBLE, 0, comm);
      MPI_Scatter(A, local_m*n, MPI_DOUBLE, local_D, local_m*n, MPI_DOUBLE, 0, comm);
   }
}  /* Read_matrix */
/*-------------------------------------------------------------------*/
void Print_matrix(double local_A[], int m, int local_n, 
      int n, MPI_Datatype col, int my_rank, MPI_Comm comm) {
   double* A = NULL;
   int i, j;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));

      MPI_Gather(local_A, m*local_n, MPI_DOUBLE, A, 1, col, 0, comm);

      printf("The matrix is:\n");
      for (i = 0; i < m; i++) {
         for (j = 0; j < n; j++)
            printf("%.2f ", A[i*n+j]);
         printf("\n");
      }
      
      free(A);
   } else {
      MPI_Gather(local_A, m*local_n, MPI_DOUBLE, A, 1, col, 0, comm);
   }
}  /* Print_matrix */
/*-------------------------------------------------------------------*/
void Read_vector(double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm) {
   double* vec = NULL;
   int i;
   
   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      
      for (i = 0; i < n; i++)
         // scanf("%lf", &vec[i]);
       vec[i] = i+1;
      MPI_Scatter(vec, local_n, MPI_DOUBLE,
               local_vec, local_n, MPI_DOUBLE, 0, comm);
      free(vec);
   } else {
      MPI_Scatter(vec, local_n, MPI_DOUBLE,
               local_vec, local_n, MPI_DOUBLE, 0, comm);
   }
}  /* Read_vector */
/*-------------------------------------------------------------------*/
void Mat_vect_mult_colunas(double local_A[], double local_B[], 
      double local_C[], int local_m, int m, int n, int local_n, 
      int comm_sz, MPI_Comm comm, int my_rank) {
   
   double* my_y;
   int* recv_counts;
   int i, loc_j;
   
   recv_counts = malloc(comm_sz*sizeof(int));
   my_y = malloc(n*sizeof(double));
 
   for (i = 0; i < m ; i++) {
      my_y[i] = 0.0;
      for (loc_j = 0; loc_j < local_n ; loc_j++)
         my_y[i] += local_A[i*local_n + loc_j]*local_B[loc_j];
   }
   
   for (i = 0; i < comm_sz; i++) {
      recv_counts[i] = local_m;
   }
   
   MPI_Reduce_scatter(my_y, local_C, recv_counts, MPI_DOUBLE, MPI_SUM, comm);
   
   free(my_y);
}  
/*-------------------------------------------------------------------*/
void Print_vector(double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm) {
   double* vec = NULL;
   int i;
   
   if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
               vec, local_n, MPI_DOUBLE, 0, comm);
      for (i = 0; i < n; i++)
         printf("%f ", vec[i]);
      printf("\n");
      free(vec);
   }  else {
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,
               vec, local_n, MPI_DOUBLE, 0, comm);
   }
}  /* Print_vector */
/*-------------------------------------------------------------------*/
void Mat_vect_mult_row(double local_D[], double local_B[], 
         double local_E[], int local_m, int n, int local_n, MPI_Comm comm) {
   double* A;
   int local_i, j;

   A = malloc(n*sizeof(double));
   MPI_Allgather(local_B, local_n, MPI_DOUBLE,
         A, local_n, MPI_DOUBLE, comm);

   for (local_i = 0; local_i < local_m; local_i++) {
      local_E[local_i] = 0.0;
      for (j = 0; j < n; j++)
         local_E[local_i] += local_D[local_i*n+j]*A[j];
   }
   free(A);
}  /* Mat_vect_mult_row */