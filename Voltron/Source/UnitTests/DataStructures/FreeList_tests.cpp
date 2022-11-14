#include "DataStructures/FreeList.h"

#include <boost/test/unit_test.hpp>
#include <string>

BOOST_AUTO_TEST_SUITE(DataStructures)

BOOST_AUTO_TEST_SUITE(Kedyk)

BOOST_AUTO_TEST_SUITE(FreeList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  FreeList<std::string> fl {};

  BOOST_TEST(fl.get_block_size() == 32);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsForSmallBlockSize)
{
  FreeList<std::string> fl {4};

  BOOST_TEST(fl.get_block_size() == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateAllocates)
{
  FreeList<int> fl {8};
  //int* allocated {fl.allocate()};
}

BOOST_AUTO_TEST_SUITE_END() // FreeList_tests
BOOST_AUTO_TEST_SUITE_END() // Kedyk

BOOST_AUTO_TEST_SUITE_END() // DataStructures