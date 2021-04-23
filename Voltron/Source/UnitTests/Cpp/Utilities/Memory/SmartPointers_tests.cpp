//------------------------------------------------------------------------------
/// \file SmartPointers_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <memory>
#include <string>
#include <utility>

using std::make_unique;
using std::move;
using std::string;
using std::unique_ptr;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Memory)
BOOST_AUTO_TEST_SUITE(SmartPointers_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RawPointersCanPointToUniquePointerObjectWithGet)
{
  unique_ptr<string> uniq_str_ptr {make_unique<string>("gatcagtaa")};

  string* str_ptr {nullptr};

  str_ptr = uniq_str_ptr.get();

  BOOST_TEST(str_ptr->at(0) == 'g');
  BOOST_TEST(str_ptr->at(1) == 'a');
  BOOST_TEST(str_ptr->at(2) == 't');
  BOOST_TEST(str_ptr->at(3) == 'c');
  BOOST_TEST(str_ptr->at(4) == 'a');
  BOOST_TEST(uniq_str_ptr->at(0) == 'g');
  BOOST_TEST(uniq_str_ptr->at(1) == 'a');
  BOOST_TEST(uniq_str_ptr->at(2) == 't');
  BOOST_TEST(uniq_str_ptr->at(3) == 'c');
  BOOST_TEST(uniq_str_ptr->at(4) == 'a');

  unique_ptr<string> other_unique_str_ptr {move(uniq_str_ptr)};

  BOOST_TEST(!static_cast<bool>(uniq_str_ptr));
  BOOST_TEST(static_cast<bool>(other_unique_str_ptr));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UniquePtrAsCharPointer)
{
  


  unique_ptr<char> uniq_char_ptr;
  uniq_char_ptr.reset(cp);

  BOOST_TEST(uniq_char_ptr.get()[0] == 'I');

}

BOOST_AUTO_TEST_SUITE_END() // SmartPointers_tests
BOOST_AUTO_TEST_SUITE_END() // Memory
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp