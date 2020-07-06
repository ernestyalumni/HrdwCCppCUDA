//------------------------------------------------------------------------------
/// \file StateMonadExamples.cpp
/// \author Ernest Yeung
/// \brief
/// \ref Edward Ashford Lee, Sanjit Arunkumar Seshia.
/// Introduction to Embedded Systems: A Cyber-Physical Systems Approach
/// (The MIT Press) Second Edition. The MIT Press; Second edition
/// (December 30, 2016). ISBN-10: 0262533812. ISBN-13: 978-0262533812
///-----------------------------------------------------------------------------
#include "StateMonadExamples.h"

#include <utility>

namespace Categories
{
namespace Monads
{
namespace StateMonad
{
namespace Examples
{

namespace GarageCounter
{

GarageCounterMorphism::InternalHom::InternalHom(const InputPorts& input_ports):
	inputs_{input_ports}
{}

GarageCounterMorphism::InternalHom GarageCounterMorphism::operator()(
	const InputPorts& input_ports)
{
	return InternalHom{input_ports};
}

std::pair<unsigned int, unsigned int>
	GarageCounterMorphism::InternalHom::operator()(const unsigned int state)
{
	unsigned int final_state {state};
	int car_changes {0};

	if (inputs_.up_port_ == SignalPresence::present)
	{
		++car_changes;
	}

	if (inputs_.down_port_ == SignalPresence::present)
	{
		--car_changes;
	}

	if (car_changes < 0 && final_state < 1)
	{
		final_state = 0;
	}
	else
	{
		final_state += car_changes;
	}

	return std::make_pair<unsigned int, unsigned int>(
		std::move(final_state), std::move(final_state));
}

GarageCounterMorphism1::GarageCounterMorphism1(const InputPorts& input_ports):
	inputs_{input_ports}
{}

std::pair<unsigned int, unsigned int> GarageCounterMorphism1::operator()(
	const unsigned int state)
{
	unsigned int final_state {state};
	int car_changes {0};

	if (inputs_.up_port_ == SignalPresence::present)
	{
		++car_changes;
	}

	if (inputs_.down_port_ == SignalPresence::present)
	{
		--car_changes;
	}

	if (car_changes < 0 && final_state < 1)
	{
		final_state = 0;
	}
	else
	{
		final_state += car_changes;
	}

	return std::make_pair<unsigned int, unsigned int>(
		std::move(final_state), std::move(final_state));
}

} // namespace GarageCounter

namespace ModestThermostat
{

InternalHom::InternalHom(
	const double heat_on_limit,
	const double heat_off_limit,
	const double temperature
	):
	heat_on_maximum_temperature_{heat_on_limit},
	heat_off_minimum_temperature_{heat_off_limit},
	temperature_{temperature}
{}

std::pair<State, InternalHom::OptionalOutput> InternalHom::operator()(
	const State state)
{
	State final_state {state};
	// Avoids chattering, where heater would turn on and off rapidly when
	// temperature is close to setpoint temperature.
	//if (temperature_ > heat_on_maximum_temperature_) ||
	//	(temperature_ < heat_off_minimum_temperature_)
	//{
	//	return std::make_pair<State, OptionalOutput>(state, std::nullopt);
	//}

	if ((temperature_ <= heat_on_maximum_temperature_) &&
		(state == State::cooling))
	{
		return std::make_pair<State, OptionalOutput>(
			State::heating,
			std::make_optional<Output>(Output::heatOn));
	}

	if ((temperature_ >= heat_off_minimum_temperature_) &&
		(state == State::heating))
	{
		return std::make_pair<State, OptionalOutput>(
			State::cooling,
			std::make_optional<Output>(Output::heatOff));
	}

	return std::make_pair<State, OptionalOutput>(
		//std::forward<State>(state),
		std::move(final_state),
		std::nullopt);
}

ModestThermostatMorphism::ModestThermostatMorphism(
	const double heat_on_limit,
	const double heat_off_limit
	):
	heat_on_maximum_temperature_{heat_on_limit},
	heat_off_minimum_temperature_{heat_off_limit}
{}

InternalHom ModestThermostatMorphism::operator()(const double temperature)
{
	return InternalHom{
		heat_on_maximum_temperature_,
		heat_off_minimum_temperature_,
		temperature};
}

} // namespace ModestThermostat

namespace TrafficLight
{

InternalHom::InternalHom(
	const unsigned int count,
	const Pedestrian pedestrian,
	const unsigned int count_time,
	const unsigned int yellow_to_red_time
	):
	count_{count},
	pedestrian_{pedestrian},
	count_time_{count_time},
	yellow_to_red_time_{yellow_to_red_time}
{}

std::pair<State, Output> InternalHom::operator()(
	const State state)
{
	State final_state {state};

	// It starts in red state and counts for 60 seconds.
	if ((state == State::red) && (count_ >= count_time_))
	{
		Output output {std::make_optional<Signal>(Signal::signal_green), true};

		return std::make_pair<State, Output>(
			State::green,
			std::move(output));
	}

	// It had transitioned to green, where it'll remain until pure input
	// pedestrian is present.

	if ((state == State::green) && (pedestrian_ == Pedestrian::present))
	{

		// When pedestrian is present, machine transitions to yellow if it has been
		// in state green for at least 60 seconds.
		if (count_ >= count_time_)
		{
			final_state = State::yellow;

			Output output {std::make_optional<Signal>(Signal::signal_yellow), true};

			return std::make_pair<State, Output>(
				std::move(final_state),
				std::move(output));
		}
		// Otherwise, it transitions to pending, where it stays for remainder of 
		// 60 seconds.
		else
		{
			final_state = State::pending;

			Output output {std::nullopt, false};

			return std::make_pair<State, Output>(
				std::move(final_state),
				std::move(output));
		}
	}

	if ((state == State::pending) && (count_ >= count_time_))
	{
		Output output {std::make_optional<Signal>(Signal::signal_yellow), true};

		return std::make_pair<State, Output>(State::yellow, std::move(output));
	}

	if ((state == State::yellow) && (count_ >= yellow_to_red_time_))
	{
		Output output {std::make_optional<Signal>(Signal::signal_red), true};

		return std::make_pair<State, Output>(State::red, std::move(output));
	}

	// Lee and Seshia (2016), pp. 62, Fig. 3.10 failed to address this case:
	if ((state == State::green) && (count_ >= count_time_))
	{
		Output output {std::make_optional<Signal>(Signal::signal_yellow), true};

		return std::make_pair<State, Output>(State::yellow, std::move(output));
	}

	Output output {std::nullopt, false};

	return std::make_pair<State, Output>(
		std::move(final_state),
		std::move(output));

	//if ((state == State ))
}

TrafficLightMorphism::TrafficLightMorphism(
	const unsigned int count_time,
	const unsigned int yellow_to_red_time
	):
	count_time_{count_time},
	yellow_to_red_time_{yellow_to_red_time}
{}

InternalHom TrafficLightMorphism::operator()(
	const unsigned int count,
	const Pedestrian pedestrian)
{
	return InternalHom{
		count,
		pedestrian,
		count_time_,
		yellow_to_red_time_
	};
}

} // namespace TrafficLight

} // namespace Examples

} // namespace StateMonad
} // namespace Monads
} // namespace Categories