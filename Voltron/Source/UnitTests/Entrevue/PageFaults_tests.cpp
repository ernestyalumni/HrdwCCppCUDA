#include "QuestionsDEntrevue/PageFaults.h"

#include <boost/test/unit_test.hpp>

using QuestionsDEntrevue::page_faults;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(PageFaults_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Demonstrate)
{
  int pages[] = {1, 2, 1, 4, 2, 3, 5};
  const int n {7};
  const int c {3};

  BOOST_TEST(page_faults(n, c, pages), 5);
}

BOOST_AUTO_TEST_SUITE_END() // PageFaults
BOOST_AUTO_TEST_SUITE_END() // Entrevue