#include "DataStructures/Stacks/DynamicStack.h"

#include <boost/test/unit_test.hpp>

using DataStructures::Stacks::AsHierarchy::DynamicStack;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Stacks)
BOOST_AUTO_TEST_SUITE(DynamicStack_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  DynamicStack<int> ds {};

  BOOST_TEST(ds.size() == 0);
  BOOST_TEST(ds.is_empty());  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithSize)
{
  DynamicStack<int> ds {7};

  BOOST_TEST(ds.size() == 0);
  BOOST_TEST(ds.is_empty());    
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(WorksAsStack)
{
  DynamicStack<int> ds {7};

  ds.push(15);
  ds.push(6);
  ds.push(2);
  ds.push(9);

  BOOST_TEST(ds.size() == 4);
  BOOST_TEST(!ds.is_empty());
  BOOST_TEST(ds.top() == 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TopWorks)
{
  DynamicStack<int> ds {7};

  ds.push(15);
  ds.push(6);
  ds.push(2);
  ds.push(9);
  ds.push(17);
  ds.push(3);

  BOOST_TEST(ds.size() == 6);
  BOOST_TEST(!ds.is_empty());
  BOOST_TEST(ds.top() == 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PopWorks)
{
  DynamicStack<int> ds {7};

  ds.push(15);
  ds.push(6);
  ds.push(2);
  ds.push(9);
  ds.push(17);
  ds.push(3);

  BOOST_TEST(ds.pop() == 3);

  BOOST_TEST(ds.size() == 5);
  BOOST_TEST(!ds.is_empty());
  BOOST_TEST(ds.top() == 17);
}

BOOST_AUTO_TEST_SUITE_END() // DynamicStack_tests
BOOST_AUTO_TEST_SUITE_END() // Stacks
BOOST_AUTO_TEST_SUITE_END() // DataStructures