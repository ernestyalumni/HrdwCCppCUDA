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
using Categories::Monads::StateMonad::Compose;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(StateMonadExamples_tests)

BOOST_AUTO_TEST_SUITE(GarageCounter_tests)

const GarageCounter::InputPorts up_not_down {
	GarageCounter::SignalPresence::present,
	GarageCounter::SignalPresence::absent};

const GarageCounter::InputPorts not_up_down {
	GarageCounter::SignalPresence::absent,
	GarageCounter::SignalPresence::present};

// Example 3.4, Fig. 3.4, garage counter Finite-State Machine (FSM), Lee and
// Seshia (2016), pp. 50

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GarageCounterIncrements)
{
	GarageCounter::GarageCounterMorphism1 reaction {up_not_down};

	BOOST_TEST(reaction(5).first == 6);
	BOOST_TEST(reaction(6).first == 7);
	BOOST_TEST(reaction(7).first == 8);

	{
		GarageCounter::GarageCounterMorphism garage_counter {};
		auto reaction = garage_counter(up_not_down);

		BOOST_TEST(reaction(5).first == 6);
		BOOST_TEST(reaction(6).first == 7);
		BOOST_TEST(reaction(7).first == 8);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GarageCounterDecrements)
{
	GarageCounter::GarageCounterMorphism1 reaction {not_up_down};

	BOOST_TEST(reaction(5).first == 4);
	BOOST_TEST(reaction(6).first == 5);
	BOOST_TEST(reaction(7).first == 6);

	{
		GarageCounter::GarageCounterMorphism garage_counter {};
		auto reaction = garage_counter(not_up_down);

		BOOST_TEST(reaction(5).first == 4);
		BOOST_TEST(reaction(6).first == 5);
		BOOST_TEST(reaction(7).first == 6);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComposeGarageCounterMorphism1s)
{
	auto composed_reactions = Compose(
		GarageCounter::GarageCounterMorphism{},
		GarageCounter::GarageCounterMorphism{});

}

BOOST_AUTO_TEST_SUITE_END() // GarageCounter_tests

BOOST_AUTO_TEST_SUITE(ModestThermostatMorphism_tests)

//------------------------------------------------------------------------------
/// \ref pp. 51, Lee and Seshia, Introduction to Embedded Systems, Figure 3.5:
/// A model of a thermostat with hysteresis.
//------------------------------------------------------------------------------
ModestThermostat::ModestThermostatMorphism thermostat {18, 22};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NoChangeForTemperaturesInHysteresis)
{
	{
		auto transition_function = thermostat(18.1);
		auto result = transition_function(ModestThermostat::State::cooling);
		BOOST_TEST((result.first == ModestThermostat::State::cooling));
		BOOST_TEST(!result.second);
	}
	{
		auto transition_function = thermostat(21.9);
		auto result = transition_function(ModestThermostat::State::heating);
		BOOST_TEST((result.first == ModestThermostat::State::heating));
		BOOST_TEST(!result.second);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContinuesCoolingIfTemperatureIsAboveALimit)
{
	auto transition_function = thermostat(22.1);
	auto result = transition_function(ModestThermostat::State::cooling);
	BOOST_TEST((result.first == ModestThermostat::State::cooling));
	BOOST_TEST(!result.second);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContinuesCoolingIfTemperatureIsAtALimit)
{
	auto transition_function = thermostat(22.0);
	auto result = transition_function(ModestThermostat::State::cooling);
	BOOST_TEST((result.first == ModestThermostat::State::cooling));
	BOOST_TEST(!result.second);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContinuesHeatingIfTemperatureIsBelowALimit)
{
	auto transition_function = thermostat(17.9);
	auto result = transition_function(ModestThermostat::State::heating);
	BOOST_TEST((result.first == ModestThermostat::State::heating));
	BOOST_TEST(!result.second);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ContinuesHeatingIfTemperatureIsBelowAtALimit)
{
	auto transition_function = thermostat(18.0);
	auto result = transition_function(ModestThermostat::State::heating);
	BOOST_TEST((result.first == ModestThermostat::State::heating));
	BOOST_TEST(!result.second);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BeginsHeatingIfTemperatureIsBelowALimitAndHadBeenCooling)
{
	auto transition_function = thermostat(17.9);
	auto result = transition_function(ModestThermostat::State::cooling);
	BOOST_TEST((result.first == ModestThermostat::State::heating));
	BOOST_TEST(static_cast<bool>(result.second));
	BOOST_TEST((*result.second == ModestThermostat::Output::heatOn));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BeginsCoolingIfTemperatureIsAboveALimitAndHadBeenHeating)
{
	auto transition_function = thermostat(22.1);
	auto result = transition_function(ModestThermostat::State::heating);
	BOOST_TEST((result.first == ModestThermostat::State::cooling));
	BOOST_TEST(static_cast<bool>(result.second));
	BOOST_TEST((*result.second == ModestThermostat::Output::heatOff));
}

BOOST_AUTO_TEST_SUITE_END() // ModestThermostatMorphism_tests

BOOST_AUTO_TEST_SUITE(TrafficLight_tests)

//------------------------------------------------------------------------------
/// \ref pp. 62, Lee and Seshia, Introduction to Embedded Systems, Figure 3.10:
/// State machine model of traffic light controller that keeps track of passage,
/// assuming it reacts at regular intervals.
//------------------------------------------------------------------------------
constexpr unsigned int count_time {60};
constexpr unsigned int yellow_to_red_time {5};
TrafficLight::TrafficLightMorphism traffic_light {
	count_time,
	yellow_to_red_time};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
	TransitionsFromRedToGreenOnlyWhenCountIsGreaterThanOrEqualToCountTime)
{
	{
		auto transition_function =
			traffic_light(59, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::red));
		BOOST_TEST(!result.second.signal_);
		BOOST_TEST(!result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(59, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::red));
		BOOST_TEST(!result.second.signal_);
		BOOST_TEST(!result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(60, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::green));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_green));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(61, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::green));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_green));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(60, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::green));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_green));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(61, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::red);
		BOOST_TEST((result.first == TrafficLight::State::green));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_green));
		BOOST_TEST(result.second.reset_count_);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GreenToYellowWhenCountIsGreaterThanOrEqualToCountTime)
{
	{
		auto transition_function =
			traffic_light(59, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::green));
		BOOST_TEST(!result.second.signal_);
		BOOST_TEST(!result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(60, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::yellow));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_yellow));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(61, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::yellow));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_yellow));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(60, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::yellow));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_yellow));
		BOOST_TEST(result.second.reset_count_);
	}
	{
		auto transition_function =
			traffic_light(61, TrafficLight::Pedestrian::absent);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::yellow));
		BOOST_TEST((
			result.second.signal_.value() == TrafficLight::Signal::signal_yellow));
		BOOST_TEST(result.second.reset_count_);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
	YellowToPendingWhenCountIsLessThanCountTimeAndPedestrianPresent)
{
	{
		auto transition_function =
			traffic_light(59, TrafficLight::Pedestrian::present);
		auto result = transition_function(TrafficLight::State::green);
		BOOST_TEST((result.first == TrafficLight::State::pending));
		BOOST_TEST(!result.second.signal_);
		BOOST_TEST(!result.second.reset_count_);
	}
}

BOOST_AUTO_TEST_SUITE_END() // TrafficLight_tests

BOOST_AUTO_TEST_SUITE_END() // StateMonadExamples_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories