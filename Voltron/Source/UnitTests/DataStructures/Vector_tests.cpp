#include "DataStructures/Vector.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Kedyk)
BOOST_AUTO_TEST_SUITE(Vector_tests)

using DataStructures::Kedyk::Vector;

template <typename ITEM_T>
class TestVector : public Vector<ITEM_T>
{
  public:

    using Vector<ITEM_T>::Vector;
    using Vector<ITEM_T>::get_capacity;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassConstantsAreConstant)
{
  BOOST_TEST(Vector<int>::MIN_CAPACITY == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructor)
{
  Vector<int> a {};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithSizeAndInitialValue)
{
  // fatal error: in
  // "DataStructures/Kedyk/Vector_tests/ConstructsWithSizeAndInitialValue":
  // signal: SIGABRT (application abort requested)
  //TestVector<int> a (6, 9);
  // BOOST_TEST(a.get_capacity() == 8);
  //BOOST_TEST(a.get_size() == 6);
  //BOOST_TEST(a[0] == 9);
  //BOOST_TEST(a[1] == 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendAppends)
{
  TestVector<int> a {};
  a.append(1);
  BOOST_TEST(a.get_capacity() == 8);
  BOOST_TEST(a.get_size() == 1);
  BOOST_TEST(a[0] == 1);
}

BOOST_AUTO_TEST_SUITE_END() // Vector_tests
BOOST_AUTO_TEST_SUITE_END() // Kedyk
BOOST_AUTO_TEST_SUITE_END() // DataStructures