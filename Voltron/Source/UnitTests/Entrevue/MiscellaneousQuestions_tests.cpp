//------------------------------------------------------------------------------
/// \file MiscellaneousQuestions_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref 
/// \details To run only these unit tests, do this:
/// ./Check --run_test="Entrevue/Miscellaneous"
//------------------------------------------------------------------------------
#include "Cpp/Utilities/SuperBitSet.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>
#include <limits>

using Cpp::Utilities::SuperBitSet;
using Cpp::Utilities::number_of_bits_in_a_byte;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(Miscellaneous)
BOOST_AUTO_TEST_SUITE(MiscellaneousQuestions_tests)

BOOST_AUTO_TEST_SUITE(Triplebyte_tests)

// From a Triplebyte advertisement.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorPrecedence)
{
  {
    int x {1};

    for (int i {0}; i < 3; ++i)
    {
      x += i * 5;  
    }

    BOOST_TEST(x == 16);
  }
  {
    int x {1};

    for (int i {0}; i < 3; i++)
    {
      x += i * 5;
    }

    BOOST_TEST(x == 16);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IntegerLiteralBitsAsChars)
{

}

BOOST_AUTO_TEST_SUITE_END() // Triplebyte_tests
BOOST_AUTO_TEST_SUITE_END() // MiscellaneousQuestions_tests
BOOST_AUTO_TEST_SUITE_END() // Miscellaneous
BOOST_AUTO_TEST_SUITE_END() // Entrevue