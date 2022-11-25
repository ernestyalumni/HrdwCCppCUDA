#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ParseTextFile.h"
#include "UnitTests/DataStructures/LinkedLists/DoublyLinkedListTestValues.h"
#include "UnitTests/FileIO/ExampleTestValues.h"

#include <boost/test/unit_test.hpp>

using CsVtEdu::InfinitePrecisionCalculator::ParseTextFile;
using CsVtEdu::InfinitePrecisionCalculator::convert_to_int;
using CsVtEdu::InfinitePrecisionCalculator::strip_leading_zeros;
using DoublyLinkedListTestValues =
  UnitTests::DataStructures::LinkedLists::DoublyLinkedListTestValues;
using UnitTests::FileIO::example_file_path_1;


BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(ParseTextFile_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  ParseTextFile parse_text_file {example_file_path_1()};

  BOOST_TEST(parse_text_file.read_in_.size() == 10);
}

BOOST_AUTO_TEST_SUITE_END() // ParseTextFile_tests

BOOST_AUTO_TEST_SUITE(StripLeadingZeroes_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StripLeadingZerosStrips)
{
  DoublyLinkedListTestValues tv {};
  tv.large_operand_1_.push_back(0);
  tv.large_operand_2_.push_back(0);
  tv.large_operand_2_.push_back(0);
  tv.large_operand_3_.push_back(0);
  tv.large_operand_3_.push_back(0);
  tv.large_operand_3_.push_back(0);

  BOOST_TEST_REQUIRE(tv.large_operand_1_.size() == 9);
  BOOST_TEST_REQUIRE(tv.large_operand_2_.size() == 7);
  BOOST_TEST_REQUIRE(tv.large_operand_3_.size() == 12);

  strip_leading_zeros<int>(tv.large_operand_1_);
  BOOST_TEST(tv.large_operand_1_.size() == 8);
  BOOST_TEST(tv.large_operand_1_.tail()->retrieve() == 9);
  BOOST_TEST(tv.large_operand_1_.tail()->previous()->retrieve() == 9);

  strip_leading_zeros<int>(tv.large_operand_2_);
  BOOST_TEST_REQUIRE(tv.large_operand_2_.size() == 5);
  BOOST_TEST(tv.large_operand_2_.tail()->retrieve() == 6);
  BOOST_TEST(tv.large_operand_2_.tail()->previous()->retrieve() == 6);

  strip_leading_zeros<int>(tv.large_operand_3_);
  BOOST_TEST_REQUIRE(tv.large_operand_3_.size() == 9);
  BOOST_TEST(tv.large_operand_3_.tail()->retrieve() == 8);
  BOOST_TEST(tv.large_operand_3_.tail()->previous()->retrieve() == 7);
}

BOOST_AUTO_TEST_SUITE_END() // StripLeadingZeros_tests

BOOST_AUTO_TEST_SUITE(ConvertToInt_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConvertToIntConvertsToInt)
{
  DoublyLinkedListTestValues tv {};
  BOOST_TEST(convert_to_int<int>(tv.large_operand_1_) == 99954321);
  BOOST_TEST(convert_to_int<int>(tv.large_operand_2_) == 66789);
  BOOST_TEST(convert_to_int<int>(tv.large_operand_3_) == 876543210);
  BOOST_TEST(convert_to_int<int>(tv.large_operand_4_) == 3456789);
}

BOOST_AUTO_TEST_SUITE_END() // ConvertToInt_tests

BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects