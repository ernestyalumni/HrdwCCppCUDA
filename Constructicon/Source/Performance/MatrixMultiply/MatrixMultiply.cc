#include "MatrixMultiply.h"

#include "Utilities/DebuggingMacros.h"

#include <cstdio>

namespace Performance
{
namespace MatrixMultiply
{

Mat::Mat(const unsigned int rows, const unsigned int columns):
  rows_{rows},
  cols_{columns},
  values_{new double*[rows]}
{
  make_matrix();
}

Mat::~Mat()
{
  free_matrix();
}

void Mat::make_matrix()
{
  for (unsigned int i {0}; i < rows_; ++i)
  {
    values_[i] = new double[cols_];
  }
}

void Mat::free_matrix()
{
  for (unsigned int i {0}; i < rows_; ++i)
  {
    delete[] values_[i];
  }

  delete[] values_;
}

int matrix_multiply_run(const Mat& A, const Mat& B, Mat& C)
{
  debugging_assert(
    A.cols_ == B.rows_,
    "A.cols_ = %d, B.rows_ = %d\n",
    A.cols_,
    B.rows_);

  for (unsigned int i {0}; i < A.rows_; ++i)
  {
    for (unsigned int j {0}; j < B.cols_; ++j)
    {
      for (unsigned int k {0}; k < A.cols_; ++k)
      {
        C.values_[i][j] += A.values_[i][k] * B.values_[k][j];
      }
    }
  }

  return 0;
}

std::ostream& operator<<(std::ostream& os, const Mat& A)
{
  for (unsigned int i {0}; i < A.rows_; ++i)
  {
    for (unsigned int j {0}; j < A.cols_; ++j)
    {
      os << A.values_[i][j] << ' ';
    }
    os << '\n';
  }
  os << '\n';

  return os;
}

// Print matrix
void print_matrix(const Mat& a)
{
  printf("------------\n");
  for (unsigned int i {0}; i < a.rows_; ++i)
  {
    for (unsigned int j {0}; j < a.cols_; ++j)
    {
      printf("  %3f  ", a.values_[i][j]);
    }
    printf("\n");
  }
  printf("------------\n");
}

} // namespace MatrixMultiply
} // namespace Performance
