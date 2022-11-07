#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ParseTextFile.h"
#include "UnitTests/FileIO/ExampleTestValues.h"

#include <boost/test/unit_test.hpp>
#include <string>

using CsVtEdu::InfinitePrecisionCalculator::ParseTextFile;
using UnitTests::FileIO::example_file_path_1;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  const std::string current_path_1 {example_file_path_1()};
  ParseTextFile parse_text_file {example_file_path_1()};

  BOOST_TEST(parse_text_file.read_in_.size() == 10);
}

BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator_tests
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects