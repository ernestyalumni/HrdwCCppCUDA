#include "ExampleTestValues.h"

#include <boost/test/unit_test.hpp>
#include <cctype> // std::isspace
#include <filesystem>
#include <fstream>
#include <string> // std::getline
#include <vector>

using UnitTests::FileIO::example_file_path_1;

BOOST_AUTO_TEST_SUITE(FileIO)

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
    bool empty_line {false};
    for (std::size_t i {0}; i < line.size(); ++i)
    {
      if (!std::isspace(line[i]))
      {
        empty_line = true;
      }
    }
    std::cout << "line: " << line << " " << line.empty() << " " << i <<
      " empty? " << empty_line << "\n";
    read_in.emplace_back(line);
    ++i;
  }

  for (auto& x : read_in)
  {
    std::cout << "line: " << x << "\n";
  }

  BOOST_TEST(true);
}


BOOST_AUTO_TEST_SUITE_END() // FileIO