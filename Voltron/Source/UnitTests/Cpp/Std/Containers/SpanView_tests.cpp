//------------------------------------------------------------------------------
/// \file SpanView_tests.cpp
//------------------------------------------------------------------------------
#include "Cpp/Std/Containers/SpanView.h"

#include <boost/test/unit_test.hpp>
#include <string>

using Cpp::Std::Containers::SpanView;
using std::string;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Std)
BOOST_AUTO_TEST_SUITE(Containers)
BOOST_AUTO_TEST_SUITE(SpanView_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(SubspanObtainsSubsetView)
{
  string test_string {"hello"};

  SpanView<char> large_span_view {
    test_string.data(),
    test_string.size()};

  BOOST_TEST(large_span_view[0] == 'h');
  BOOST_TEST(large_span_view[1] == 'e');
  BOOST_TEST(large_span_view[2] == 'l');
  BOOST_TEST(large_span_view[3] == 'l');
  BOOST_TEST(large_span_view[4] == 'o');

  BOOST_TEST(large_span_view.subspan(1, 2)[0] == 'e');
  BOOST_TEST(large_span_view.subspan(1, 2)[1] == 'l');
}

BOOST_AUTO_TEST_SUITE_END() // SpanView_tests
BOOST_AUTO_TEST_SUITE_END() // Containers
BOOST_AUTO_TEST_SUITE_END() // Std
BOOST_AUTO_TEST_SUITE_END() // Cpp