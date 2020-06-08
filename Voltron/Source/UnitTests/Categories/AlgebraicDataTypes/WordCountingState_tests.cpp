//------------------------------------------------------------------------------
/// \file WordCountingState_tests.cpp
//------------------------------------------------------------------------------
#include "Categories/AlgebraicDataTypes/WordCountingState.h"

#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using Categories::AlgebraicDataTypes::WordCounting::ProgramT;
using Categories::AlgebraicDataTypes::WordCounting::States::RunningT;

using Tools::TemporaryDirectory;

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ProgramTCounts)
{
  // cf. https://en.cppreference.com/w/cpp/io/basic_ifstream
  TemporaryDirectory temp_dir {"/temp"};  
  std::string filename {"Test.txt"};
  const std::string full_filepath {temp_dir.path() + "/" + filename};
  std::ofstream temp_out {full_filepath};
  temp_out << "As we knelt on the kitchen floor \n" <<
    "I said mommy Imma love you till you don't hurt no more \n" <<
    "And when I'm older, you aint gotta work no more \n" <<
    "And Imma get you that mansion that we couldn't afford \n" <<
    "See you're, unbreakable, unmistakable \n" <<
    "Highly capable, lady that's makin loot \n" <<
    "A livin legend too, just look at what heaven do \n" <<
    "Send us an angel, and I thank you (Hey Mama) \n";
  temp_out.close();

  std::ifstream temp_in {full_filepath};
  std::cout << temp_in.rdbuf() << '\n';
  std::ifstream temp_in2 {full_filepath};
  std::cout << temp_in2.rdbuf() << '\n';
  std::ifstream temp_in3 {full_filepath};
  std::istream_iterator<std::string> temp_iter {temp_in3};

  auto temp_distance =
    std::distance(temp_iter, std::istream_iterator<std::string>());
  std::cout << " temp distance : " << temp_distance << '\n';

  RunningT running {full_filepath};
  running.count_words();
  BOOST_TEST(running.count() == 69);

  ProgramT program;
  BOOST_TEST(program.count() == 0);  
  program.count_words(full_filepath);
  std::cout << "Program count : " << program.count() << std::endl;
  BOOST_TEST(program.count() == 69);
}


BOOST_AUTO_TEST_SUITE_END() // WordCountingState_tests
BOOST_AUTO_TEST_SUITE_END() // AlgebraicDataTypes
BOOST_AUTO_TEST_SUITE_END() // Categories