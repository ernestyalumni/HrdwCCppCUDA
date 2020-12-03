//------------------------------------------------------------------------------
// \file PeakFinding_tests.cpp
//------------------------------------------------------------------------------
#include "Algorithms/PeakFinding.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using Algorithms::PeakFinding::OneDim::straightforward_search;
using std::vector;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(PeakFinding_tests)
BOOST_AUTO_TEST_SUITE(OneDim_tests)

// cf. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf

const vector<int> a {6, 7, 4, 3, 2, 1, 4, 5};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStraightforwardAlgorithm)
{
  BOOST_TEST(straightforward_search(a.data(), a.size()) == a.size() - 1);
  BOOST_TEST(straightforward_search(a) == a.size() - 1); 
}

BOOST_AUTO_TEST_SUITE_END() // OneDim_tests
BOOST_AUTO_TEST_SUITE_END() // PeakFinding_tests
BOOST_AUTO_TEST_SUITE_END() // Algorithms