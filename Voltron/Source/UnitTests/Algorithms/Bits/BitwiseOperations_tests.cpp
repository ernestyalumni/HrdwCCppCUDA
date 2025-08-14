#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(Bits)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BitwiseAnd)
{
  //----------------------------------------------------------------------------
  /// \ref Example from simpleCUDA2GL.cu in cuda-samples of 0 Introduction,
  /// simpleCUDA2GL.
  //----------------------------------------------------------------------------
  auto get_value = [](int x) -> int { return (x & 0x20) ? 100 : 0; };

  for (int i {0}; i < 32; ++i)
  {
    BOOST_TEST(get_value(i) == 0);
  }
  for (int i {32}; i < 64; ++i)
  {
    BOOST_TEST(get_value(i) == 100);
  }
  for (int i {64}; i < 96; ++i)
  {
    BOOST_TEST(get_value(i) == 0);
  }
  for (int i {96}; i < 128; ++i)
  {
    BOOST_TEST(get_value(i) == 100);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Bits
BOOST_AUTO_TEST_SUITE_END() // Algorithms