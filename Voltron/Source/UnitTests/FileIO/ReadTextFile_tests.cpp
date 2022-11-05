#include "ExampleTestValues.h"

#include <boost/test/unit_test.hpp>
#include <cctype> // std::isspace
#include <filesystem>
#include <fstream>
#include <string> // std::getline
#include <unordered_set>
#include <vector>

using UnitTests::FileIO::example_file_path_1;

BOOST_AUTO_TEST_SUITE(FileIO)

std::unordered_set<std::size_t> empty_lines {
  0, 3, 6, 7, 8, 10, 11, 13, 15, 17, 18, 20, 21, 22, 24, 25, 32, 33, 35};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StepsToConstructFromWorkingPathWorks)
{
  // In C++, file stream classes are designed with the idea that a file should
  // simply be viewed as a stream or array of uninterpreted bytes.
  std::ifstream read_in_stream {example_file_path_1()};

  std::vector<std::string> read_in {};

  // getline(std::basic_istream<>& input, std::basic_string<>& str);
  // Extracts characters from input and appends them to str.
  // default delimiter is endline character.
  volatile bool expression_started {true};
  // Not possible to initialize with 2 different types in a for loop:
  // https://stackoverflow.com/questions/28763237/for-loop-initialization-with-two-different-variable-type
  std::size_t i {0};
  for (std::string line {}; std::getline(read_in_stream, line);)
  {
    bool is_empty_line {true};
    for (std::size_t j {0}; j < line.size(); ++j)
    {
      if (!std::isspace(line[j]))
      {
        is_empty_line = false;
      }
    }

    if (is_empty_line)
    {
      BOOST_TEST((empty_lines.find(i) != empty_lines.end()));
    }
    else
    {
      BOOST_TEST((empty_lines.find(i) == empty_lines.end()));
    }

    read_in.push_back(line);
    ++i;
  }

  for (auto iter = empty_lines.begin(); iter != empty_lines.end(); ++iter)
  {
    const std::string possible_empty_string {read_in.at(*iter)};

    bool is_empty_line {true};
    for (std::size_t i {0}; i < possible_empty_string.size(); ++i)
    {
      if (!std::isspace(possible_empty_string[i]))
      {
        is_empty_line = false;
      }
    }
    BOOST_TEST(is_empty_line);
  }

  /*
  for (auto& x : read_in)
  {
    std::cout << "line: " << x << "\n";
  }
  */

  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // FileIO