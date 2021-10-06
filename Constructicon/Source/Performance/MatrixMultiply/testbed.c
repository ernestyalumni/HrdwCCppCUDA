/*******************************************************************************
 * @ref MIT 6.172 Fall 2018 Hwk 1, matrix-multiply
 * @file testbed.c
 * @brief This file runs your code, timing its execution and printing out the
 * result.
 * @details
 * 
 * EXAMPLE_USAGE
 * =============
 *
 * If you encounter a segmentation fault, bus error, or assertion failure, or
 * you just want to set a breakpoint, use debugging tool GDB.
 * 
 * gdb --args ./matrix_multiply_testbed
 * 
 * make DEBUG=1
 * gdb --args ./matrix_multiply_testbed
 * 
 * make clean
 * make ASAN=1
 * 
 * make clean && make DEBUG=1
 * valgrind --leak-check=full ./matrix_multiply_testbed -p
 ******************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fasttime.h"
#include "matrix_multiply.h"

int main(int argc, char** argv) {
  int optchar = 0;
  int show_usec = 0;
  int should_print = 0;
  int use_zero_matrix = 0;

  // Always use the same seed, so that our tests are repeatable.
  unsigned int random_seed = 1;

  matrix* A;
  matrix* B;
  matrix* C;

  // Original value is 4.
  const int kMatrixSize = 200;

  // Parse command line arguments.

  // https://man7.org/linux/man-pages/man3/getopt.3.html
  // getopt - parses command-line.
  // If there are no more option characters, getopt() returns -1.
  // getopt() permutes contents of argv as it scans.
  // int getopt(int argc, char* const argv[], const char *optstring);
  // optstring is a string containing legitimate option characters.
  while ((optchar = getopt(argc, argv, "upz")) != -1) {
    switch (optchar) {
      case 'u':
        show_usec = 1;
        break;
      case 'p':
        should_print = 1;
        break;
      case 'z':
        use_zero_matrix = 1;
        break;
      default:
        printf("Ignoring unrecognized option: %c\n", optchar);
        continue;
    }
  }

  // This is a trick to make the memory bug leads to a wrong output.
  int size = sizeof(int) * 4;
  int* temp[20];

  for (int i = 0; i < 20; i++) {
    temp[i] = (int*)malloc(size);
    memset(temp[i], 1, size);
  }

  int total = 0;
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 4; j++) {
      total += temp[i][j];
    }
  }

  if (!total) {
    printf("Trick to stop mallocs from being optimized out.");
  }
  for (int i = 0; i < 20; i++) {
    free(temp[i]);
  }

  fprintf(stderr, "Setup\n");

  // Originally A was this line, to demonstrate using gdb for a segfault.
  //A = make_matrix(kMatrixSize, kMatrixSize + 1);
  A = make_matrix(kMatrixSize, kMatrixSize);
  B = make_matrix(kMatrixSize, kMatrixSize);
  C = make_matrix(kMatrixSize, kMatrixSize);

  if (use_zero_matrix) {
    for (int i = 0; i < A->rows; i++) {
      for (int j = 0; j < A->cols; j++) {
        A->values[i][j] = 0;
      }
    }
    for (int i = 0; i < B->rows; i++) {
      for (int j = 0; j < B->cols; j++) {
        B->values[i][j] = 0;
      }
    }
  } else {
    for (int i = 0; i < A->rows; i++) {
      for (int j = 0; j < A->cols; j++) {
        A->values[i][j] = rand_r(&random_seed) % 10;
      }
    }
    for (int i = 0; i < B->rows; i++) {
      for (int j = 0; j < B->cols; j++) {
        B->values[i][j] = rand_r(&random_seed) % 10;
      }
    }
  }

  if (should_print) {
    printf("MatrixA: \n");
    print_matrix(A);

    printf("Matrix B: \n");
    print_matrix(B);
  }

  fprintf(stderr, "Running matrix_multiply_run()...\n");

  fasttime_t time1 = gettime();
  matrix_multiply_run(A, B, C);
  fasttime_t time2 = gettime();

  fasttime_t time1_ikj = gettime();
  matrix_multiply_ikj(A, B, C);
  fasttime_t time2_ikj = gettime();

  fasttime_t time1_jik = gettime();
  matrix_multiply_jik(A, B, C);
  fasttime_t time2_jik = gettime();

  fasttime_t time1_jki = gettime();
  matrix_multiply_jki(A, B, C);
  fasttime_t time2_jki = gettime();

  fasttime_t time1_kij = gettime();
  matrix_multiply_kij(A, B, C);
  fasttime_t time2_kij = gettime();

  // Otherwise, do valgrind --leak-check=full ./matrix_multiply_testbed

  if (should_print) {
    printf("---- RESULTS ----\n");
    printf("Result: \n");
    print_matrix(C);
    printf("---- END RESULTS ----\n");
  }

  if (show_usec) {
    double elapsed = tdiff(time1, time2);
    printf(
      "Elapsed execution time: %f usec\n",
      elapsed * (1000.0 * 1000.0));
 
   elapsed = tdiff(time1_ikj, time2_ikj);
    printf(
      "Elapsed execution time ikj: %f usec\n",
      elapsed * (1000.0 * 1000.0));
 
   elapsed = tdiff(time1_jik, time2_jik);
    printf(
      "Elapsed execution time jik: %f usec\n",
      elapsed * (1000.0 * 1000.0));

   elapsed = tdiff(time1_jki, time2_jki);
    printf(
      "Elapsed execution time jki: %f usec\n",
      elapsed * (1000.0 * 1000.0));

   elapsed = tdiff(time1_kij, time2_kij);
    printf(
      "Elapsed execution time kij: %f usec\n",
      elapsed * (1000.0 * 1000.0));

  } else {
    double elapsed = tdiff(time1, time2);
    printf("Elapsed execution time: %f sec\n", elapsed);

    elapsed = tdiff(time1_ikj, time2_ikj);
    printf("Elapsed execution time ikj: %f sec\n", elapsed);

    elapsed = tdiff(time1_jik, time2_jik);
    printf("Elapsed execution time jik: %f sec\n", elapsed);

    elapsed = tdiff(time1_jki, time2_jki);
    printf("Elapsed execution time jki: %f sec\n", elapsed);

    elapsed = tdiff(time1_kij, time2_kij);
    printf("Elapsed execution time kij: %f sec\n", elapsed);
  }

  /* Exercise: Fix testbed.c by freeing these matrices, A, B, C, after use with
   * the function free_matrix. Comment out following 3 lines to see original
   * error. */

  free_matrix(A);
  free_matrix(B);
  free_matrix(C);

  return 0;
}
