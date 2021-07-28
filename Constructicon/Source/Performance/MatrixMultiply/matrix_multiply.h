/*******************************************************************************
 * @ref MIT 6.172 Fall 2018 Hwk 1, matrix-multiply
 *
 * @brief Matrix Multiply
 * 
 * @details Declarations here are your API specification.
 ******************************************************************************/

#ifndef MATRIX_MULTIPLY_H_INCLUDED
#define MATRIX_MULTIPLY_H_INCLUDED

typedef struct {
  int rows;
  int cols;
  int** values;
} matrix;

// Multiply matrix A*B, store result in C.
int matrix_multiply_run(const matrix* A, const matrix* B, matrix* C);

// Allocates a row-by-cols matrix and returns it
matrix* make_matrix(int rows, int cols);

// Frees an allocated matrix
void free_matrix(matrix* m);

// Print matrix
void print_matrix(const matrix* m);

#endif // MATRIX_MULTIPLY_H_INCLUDED