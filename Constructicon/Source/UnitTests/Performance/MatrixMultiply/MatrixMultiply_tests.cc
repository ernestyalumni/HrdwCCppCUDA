#include "Performance/MatrixMultiply/MatrixMultiply.h"

#include "gtest/gtest.h"

using Performance::MatrixMultiply::Mat;

namespace GoogleUnitTests
{
namespace Performance
{
namespace MatrixMultiply
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(MatTests, MatConstructs)
{
  {
    Mat a {3, 3};

    SUCCEED();
  }
}

} // namespace MatrixMultiply
} // namespace Performance
} // namespace GoogleUnitTests