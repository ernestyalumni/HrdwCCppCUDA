//------------------------------------------------------------------------------
/// \file StateMonadExamples_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref Edward Ashford Lee, Sanjit Arunkumar Seshia.
/// Introduction to Embedded Systems: A Cyber-Physical Systems Approach
/// (The MIT Press) Second Edition. The MIT Press; Second edition
/// (December 30, 2016). ISBN-10: 0262533812. ISBN-13: 978-0262533812
//------------------------------------------------------------------------------
#include "Categories/Monads/StateMonad.h"
#include "Categories/Monads/StateMonadExamples.h"

#include <boost/test/unit_test.hpp>

using namespace Categories::Monads::StateMonad::Examples;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(StateMonadExamples_tests)

BOOST_AUTO_TEST_SUITE(GarageCounter_tests)

const GarageCounter::InputPorts up_not_down {
	GarageCounter::SignalPresence::present,
	GarageCounter::SignalPresence::absent};

// Example 3.4, Fig. 3.4, garage counter Finite-State Machine (FSM), Lee and
// Seshia (2016), pp. 50

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GarageCounterIncrements)
{
	GarageCounter::GarageCounterMorphism reaction {up_not_down};

	BOOST_TEST(reaction(5).first == 6);
	BOOST_TEST(reaction(6).first == 7);
	BOOST_TEST(reaction(7).first == 8);
}

BOOST_AUTO_TEST_SUITE_END() // GarageCounter_tests

BOOST_AUTO_TEST_SUITE_END() // StateMonadExamples_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories