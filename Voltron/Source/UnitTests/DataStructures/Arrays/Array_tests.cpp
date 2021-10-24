#include "DataStructures/Arrays/Array.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Arrays::DWHarder::Array;
using DataStructures::Arrays::DWHarder::sum;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Arrays)

BOOST_AUTO_TEST_SUITE(DWHarder)

BOOST_AUTO_TEST_SUITE(Array_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromSizeTForCapacityWithParentheses)
{
  const Array<int> a (3);

  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromStdInitializerList)
{
  const Array<int> a {54, 25, 37};

  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.size() == 3);

  BOOST_TEST(a[0] == 54);
  BOOST_TEST(a[1] == 25);
  BOOST_TEST(a[2] == 37);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendInsertsValues)
{
  Array<int> a (3);
  Array<int> b (5);

  a.append(54);
  a.append(25);
  a.append(37);

  BOOST_TEST(a[0] == 54);
  BOOST_TEST(a[1] == 25);
  BOOST_TEST(a[2] == 37);

  b.append(92);
  b.append(82);

  BOOST_TEST(b[0] == 92);
  BOOST_TEST(b[1] == 82);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BeginWorksAsAnIterator)
{
  Array<int> a {1, 2, 3, 4, 5};

  int value {1};
  for (auto iter {a.begin()}; iter < a.end(); ++iter)
  {
    BOOST_TEST(*iter == value);

    ++value;
  }
}



BOOST_AUTO_TEST_SUITE_END() // Array_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SumComputes)
{
  Array<int> a {54, 25, 37};

  BOOST_TEST(sum(a) == 116);
}

BOOST_AUTO_TEST_SUITE_END() // DWHarder

BOOST_AUTO_TEST_SUITE_END() // Arrays
BOOST_AUTO_TEST_SUITE_END() // DataStructures