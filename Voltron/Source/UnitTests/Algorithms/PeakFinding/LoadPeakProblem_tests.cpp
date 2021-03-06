//------------------------------------------------------------------------------
/// \file LoadPeakProblem_tests.cpp
/// \brief Load and parse peak problem tests.
//------------------------------------------------------------------------------
#include "Tools/Filepaths.h"
#include "Algorithms/PeakFinding/LoadPeakProblem.h"

#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream>
#include <string> // std::getline

#include <iostream>

namespace fs = std::filesystem;

using Algorithms::PeakFinding::LoadPeakProblem;
using Tools::get_data_directory;
using std::ifstream;
using std::string;

class TestLoadPeakProblem : public LoadPeakProblem
{
  public:
    using LoadPeakProblem::get_row_boundaries;
    using LoadPeakProblem::parse_first_equal_sign;
    using LoadPeakProblem::simple_parse_row;
};

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

  TestLoadPeakProblem loader;

  if (fs::exists(fs::path(file_path)))
  {
    ifstream input_file_stream {file_path};

    BOOST_TEST_REQUIRE(static_cast<bool>(input_file_stream));

    /*

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

      const auto result = loader.parse_first_equal_sign(temp_line);

      if (result)
      {
        std::cout << *result << "\n";
      }

      if (!result)
      {
        const auto resulting_boundaries = loader.get_row_boundaries(temp_line);

        if (resulting_boundaries)
        {
          const auto parsed_result =
            loader.simple_parse_row(temp_line, *resulting_boundaries);

          for (auto ij : parsed_result)
          {
            std::cout << ij;
          }

          std::cout << "\n";
        }
      }
    }
    */

    loader.parse(input_file_stream);

    BOOST_TEST(loader.number_of_rows() == 11);
    BOOST_TEST(loader.ith_row_size(0) == 11);

    BOOST_TEST(loader.get(0, 0) == 4);
    BOOST_TEST(loader.get(0, 1) == 5);
    BOOST_TEST(loader.get(0, 2) == 6);
    BOOST_TEST(loader.get(1, 0) == 5);
    BOOST_TEST(loader.get(1, 1) == 6);
    BOOST_TEST(loader.get(1, 2) == 7);
    BOOST_TEST(loader.get(2, 0) == 6);
    BOOST_TEST(loader.get(2, 1) == 7);
    BOOST_TEST(loader.get(2, 2) == 8);
  }

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // LoadPeakProblem_tests
BOOST_AUTO_TEST_SUITE_END() // PeakFinding
BOOST_AUTO_TEST_SUITE_END() // Algorithms