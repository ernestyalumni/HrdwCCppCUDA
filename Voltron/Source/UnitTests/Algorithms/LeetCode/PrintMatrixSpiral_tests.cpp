//------------------------------------------------------------------------------
/// \ref https://leetcode.com/problems/spiral-matrix/
//------------------------------------------------------------------------------
#include "Algorithms/LeetCode/PrintMatrixSpiral.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::PrintMatrixSpiral;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(PrintMatrixSpiral_tests)

const vector<vector<int>> single_row {{42, 43, 44}};
const vector<vector<int>> single_column {{42}, {43}, {44}};
const vector<vector<int>> smallest_square {{42, 43}, {45, 44}};
const vector<vector<int>> smallest_two_row_rectangle {
  {42, 43, 44},
  {47, 46, 45}};
const vector<vector<int>> smallest_two_column_rectangle {
  {42, 43},
  {47, 44},
  {46, 45}};

const vector<vector<int>> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

const vector<vector<int>> example_2 {
  {1, 2, 3, 4},
  {5, 6, 7, 8},
  {9, 10, 11, 12}};

// cf. https://leetcode.com/submissions/detail/507365756/
const vector<vector<int>> test_case_15 {
  {1, 2, 3, 4},
  {5, 6, 7, 8},
  {9, 10, 11, 12},
  {13, 14, 15, 16}};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckSingleRowOnGetLinearSpiralOrder)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(single_row)};
  BOOST_TEST(result.size() == 3);

  for (int i {0}; i < result.size(); ++i)
  {
    BOOST_TEST(result[i] == i + 42);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckSingleColumnOnGetLinearSpiralOrder)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(single_column)};
  BOOST_TEST(result.size() == 3);

  for (int i {0}; i < result.size(); ++i)
  {
    BOOST_TEST(result[i] == i + 42);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LinearSpiralOrderOn2x3)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(
    smallest_two_row_rectangle)};
  BOOST_TEST(result.size() == 6);

  for (int i {0}; i < result.size(); ++i)
  {
    BOOST_TEST(result[i] == i + 42);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LinearSpiralOrderOn3x2)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(
    smallest_two_column_rectangle)};
  BOOST_TEST(result.size() == 6);

  for (int i {0}; i < result.size(); ++i)
  {
    BOOST_TEST(result[i] == i + 42);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetLinearSpiralOrder)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(a)};

  BOOST_TEST(result.size() == 9);

  for (int i {0}; i < a.size(); ++i)
  {
    BOOST_TEST(result[i] == i + 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetLinearSpiralOrderOnExample2)
{
  PrintMatrixSpiral<int> printer;

  const vector<int> result {printer.get_linear_spiral_order(example_2)};

  BOOST_TEST(result.size() == 12);

  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  BOOST_TEST(result[2] == 3);
  BOOST_TEST(result[3] == 4);
  BOOST_TEST(result[4] == 8);
  BOOST_TEST(result[5] == 12);
  BOOST_TEST(result[6] == 11);
  BOOST_TEST(result[7] == 10);
  BOOST_TEST(result[8] == 9);
  BOOST_TEST(result[9] == 5); //
  BOOST_TEST(result[10] == 6);
  BOOST_TEST(result[11] == 7);
}

BOOST_AUTO_TEST_SUITE_END() // PrintMatrixSpiral_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms