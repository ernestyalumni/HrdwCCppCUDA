//------------------------------------------------------------------------------
/// \file Span_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Span is a class template and belongs in Containers library.
/// \ref https://en.cppreference.com/w/cpp/container/span
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <iostream>

// https://en.cppreference.com/w/cpp/preprocessor/include
// Since C++17, use __has_include, preprocessor constant expression that
// evaluates to 1 if filename found, 0 if not. Program is ill-formed if argument
// would not be a valid argument to #include directive.
//
// More refs:
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
// https://en.cppreference.com/w/cpp/utility/feature_test

#if __has_include(<span>)

#include <span>
#define have_span 1

#else
#define have_span 0

//std::cout << "\n Does not have <span> \n;"

#endif 

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Span_tests)

#if have_span == 1

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdSpanConstructsWithCharArrays)
{
  const char* kings[] = {"Antigonus", "Seleucus", "Ptolemy"};  

  const auto kings_span = std::span{kings};
}

#else

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DoesNotHaveSpan)
{
  std::cout << "\n Does not have span; have_span: " << have_span << "\n";

  BOOST_TEST(true);  
}

#endif

BOOST_AUTO_TEST_SUITE_END() // Span_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp
