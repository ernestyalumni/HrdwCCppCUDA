//------------------------------------------------------------------------------
/// \file WordCountingState_tests.cpp
//------------------------------------------------------------------------------
#include "Categories/AlgebraicDataTypes/WordCountingState.h"

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using Categories::AlgebraicDataTypes::WordCounting::ProgramT;
using Categories::AlgebraicDataTypes::WordCounting::States::RunningT;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(AlgebraicDataTypes)
BOOST_AUTO_TEST_SUITE(WordCountingState_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StatesConstructs)
{
  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RunningTCountsWords)
{
  {
    const std::string& file_name {"main.cpp"};
    std::ifstream file_eg {file_name};
    auto count {
      std::distance(
        std::istream_iterator<std::string>(file_eg),
        std::istream_iterator<std::string>())};
    
    std::cout << count << std::endl;    
  }

  RunningT running {"WordCountingState_tests.cpp"};
  running.count_words();
  std::cout << running.count() << std::endl;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProgramTCountsWords)
{
  ProgramT program;
  program.count_words("main.cpp");
  std::cout << "Program count : " << program.count() << std::endl;
  BOOST_TEST(program.count() == 0);
}


BOOST_AUTO_TEST_SUITE_END() // WordCountingState_tests
BOOST_AUTO_TEST_SUITE_END() // AlgebraicDataTypes
BOOST_AUTO_TEST_SUITE_END() // Categories