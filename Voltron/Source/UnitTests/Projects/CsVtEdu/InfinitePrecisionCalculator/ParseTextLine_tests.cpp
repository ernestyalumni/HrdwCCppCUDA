#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ParseTextFile.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ParseTextLine.h"
#include "UnitTests/FileIO/ExampleTestValues.h"

#include <boost/test/unit_test.hpp>

using CsVtEdu::InfinitePrecisionCalculator::ParseTextFile;
using CsVtEdu::InfinitePrecisionCalculator::ParseTextLine;
using UnitTests::FileIO::example_file_path_1;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(ParseTextLine_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  const std::string current_path_1 {example_file_path_1()};
  ParseTextFile parse_text_file {example_file_path_1()};
  BOOST_TEST_REQUIRE(parse_text_file.read_in_.size() == 10);

  {
    ParseTextLine parse_text_line {parse_text_file.read_in_[0]};

    BOOST_TEST(parse_text_line.read_in_.size() == 9);

    BOOST_TEST(parse_text_line.read_in_[0] == "56669777");
    BOOST_TEST(parse_text_line.read_in_[1] == "99999911111");
    BOOST_TEST(parse_text_line.read_in_[2] == "+");
    BOOST_TEST(parse_text_line.read_in_[3] == "352324012");
    BOOST_TEST(parse_text_line.read_in_[4] == "+");
    BOOST_TEST(parse_text_line.read_in_[5] == "3");
    BOOST_TEST(parse_text_line.read_in_[6] == "^");
    BOOST_TEST(parse_text_line.read_in_[7] == "555557778");
    BOOST_TEST(parse_text_line.read_in_[8] == "*");
  }
  {
    ParseTextLine parse_text_line {parse_text_file.read_in_[1]};
    BOOST_TEST(parse_text_line.read_in_.size() == 11);

    BOOST_TEST(parse_text_line.read_in_[0] == "99999999");
    BOOST_TEST(parse_text_line.read_in_[1] == "990001");
    BOOST_TEST(parse_text_line.read_in_[2] == "*");
    BOOST_TEST(parse_text_line.read_in_[3] == "1119111");
    BOOST_TEST(parse_text_line.read_in_[4] == "55565");
    BOOST_TEST(parse_text_line.read_in_[5] == "33333");
    BOOST_TEST(parse_text_line.read_in_[6] == "+");
    BOOST_TEST(parse_text_line.read_in_[7] == "*");
    BOOST_TEST(parse_text_line.read_in_[8] == "+");
    BOOST_TEST(parse_text_line.read_in_[9] == "88888888");
    BOOST_TEST(parse_text_line.read_in_[10] == "+");
  }
  {
    ParseTextLine parse_text_line {parse_text_file.read_in_[9]};
    BOOST_TEST(parse_text_line.read_in_.size() == 3);

    BOOST_TEST(parse_text_line.read_in_[0] == "2");
    BOOST_TEST(parse_text_line.read_in_[1] == "96");
    BOOST_TEST(parse_text_line.read_in_[2] == "^");
  }
}

BOOST_AUTO_TEST_SUITE_END() // ParseTextLine_tests

BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects