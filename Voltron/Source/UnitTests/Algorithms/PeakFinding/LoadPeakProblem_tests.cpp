//------------------------------------------------------------------------------
/// \file LoadPeakProblem_tests.cpp
/// \brief Load and parse peak problem tests.
//------------------------------------------------------------------------------
#include "Tools/Filepaths.h"

#include <boost/test/unit_test.hpp>
#include <fstream>

using Tools::get_data_directory;
using std::ifstream;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(PeakFinding)
BOOST_AUTO_TEST_SUITE(LoadPeakProblem_tests)

// cf. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PutProblemFileIntoInputFileStream)
{
  const auto file_path = get_data_directory().concat(
    "/Algorithms/PeakFinding/problem.py");

  ifstream input_file_stream {file_path};

  BOOST_TEST(true);

}

BOOST_AUTO_TEST_SUITE_END() // LoadPeakProblem_tests
BOOST_AUTO_TEST_SUITE_END() // PeakFinding
BOOST_AUTO_TEST_SUITE_END() // Algorithms