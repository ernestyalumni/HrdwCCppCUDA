//------------------------------------------------------------------------------
/// \file LoadPeakProblem_tests.cpp
/// \brief Load and parse peak problem tests.
//------------------------------------------------------------------------------
#include "Tools/Filepaths.h"

#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream>
#include <string> // std::getline

#include <iostream>

namespace fs = std::filesystem;

using Tools::get_data_directory;
using std::ifstream;
using std::string;

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(PeakFinding)
BOOST_AUTO_TEST_SUITE(LoadPeakProblem_tests)

// cf. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf

const string problem_path {"/Algorithms/PeakFinding/problem.py"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PutProblemFileIntoInputFileStream)
{
  const auto file_path = get_data_directory().concat(problem_path);

  if (fs::exists(fs::path(file_path)))
  {
    ifstream input_file_stream {file_path};

    BOOST_TEST_REQUIRE(static_cast<bool>(input_file_stream));

    string line;

    // cf. https://en.cppreference.com/w/cpp/string/basic_string/getline

    //while(getline)
    for (string temp_line; getline(input_file_stream, temp_line);)
    {
      std::cout << " temp_line: " << temp_line << "\n";
    }

    // cf. http://www.cplusplus.com/forum/beginner/60319/
    // Might need to call .clear() to clear any error flags after.
    input_file_stream.close();
    input_file_stream.clear();
    input_file_stream.open(file_path);

    for (string temp_line; getline(input_file_stream, temp_line);)
    {
      std::cout << " temp_line: " << temp_line << "\n";

      std::cout << temp_line.find('=') << "\n";

    }


  }


  BOOST_TEST(true);

}

BOOST_AUTO_TEST_SUITE_END() // LoadPeakProblem_tests
BOOST_AUTO_TEST_SUITE_END() // PeakFinding
BOOST_AUTO_TEST_SUITE_END() // Algorithms