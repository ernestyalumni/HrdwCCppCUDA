#ifndef PERFORMANCE_MATRIX_MULTIPLY_MATRIX_MULTIPLY_H
#define PERFORMANCE_MATRIX_MULTIPLY_MATRIX_MULTIPLY_H

#include <ostream>

namespace Performance
{
namespace MatrixMultiply
{

class Mat
{
  public:

    Mat() = delete;

    Mat(const unsigned int rows, const unsigned int columns);

    virtual ~Mat();

    // Multiply matrix A*B, store result in C.
    friend int matrix_multiply_run(const Mat& A, const Mat& B, Mat& C);

    friend std::ostream& operator<<(std::ostream& os, const Mat& A);

    // Print Matrix.
    friend void print_matrix(const Mat& m);   

  protected:

    void make_matrix();

    void free_matrix();

  private:

    unsigned int rows_;
    unsigned int cols_;

    double** values_;
};


} // namespace MatrixMultiply
} // namespace Performance

#endif // PERFORMANCE_MATRIX_MULTIPLY_MATRIX_MULTIPLY_H
