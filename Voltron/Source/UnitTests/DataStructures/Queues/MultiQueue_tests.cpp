#include "DataStructures/Queues/MultiQueue.h"

#include <boost/test/unit_test.hpp>
#include <string>

using DataStructures::Queues::DWHarder::MultiQueue;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Queues)
BOOST_AUTO_TEST_SUITE(MultiQueue_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
  {
    MultiQueue<int, 3> mq {};
  }
  {
    MultiQueue<std::string, 4> mq {};
  }
}

BOOST_AUTO_TEST_SUITE_END() // MultiQueues
BOOST_AUTO_TEST_SUITE_END() // Queues
BOOST_AUTO_TEST_SUITE_END() // DataStructures