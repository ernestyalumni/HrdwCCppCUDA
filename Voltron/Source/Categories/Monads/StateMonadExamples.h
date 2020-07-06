//------------------------------------------------------------------------------
/// \file StateMonadExamples.h
/// \author Ernest Yeung
/// \brief
/// \ref Edward Ashford Lee, Sanjit Arunkumar Seshia.
/// Introduction to Embedded Systems: A Cyber-Physical Systems Approach
/// (The MIT Press) Second Edition. The MIT Press; Second edition
/// (December 30, 2016). ISBN-10: 0262533812. ISBN-13: 978-0262533812
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_STATE_MONAD_EXAMPLES_H
#define CATEGORIES_MONADS_STATE_MONAD_EXAMPLES_H

#include <optional>
#include <utility> // std::pair

namespace Categories
{
namespace Monads
{
namespace StateMonad
{
namespace Examples
{

// Example 3.4, Fig. 3.4, garage counter Finite-State Machine (FSM), Lee and
// Seshia (2016), pp. 50

namespace GarageCounter
{

enum class SignalPresence : unsigned char
{
	absent = 0b00,
	present = 0b01
};

//using OptionalSignalPresence = std::optional<SignalPresence>;

struct InputPorts
{
//	OptionalSignalPresence up_port_;
//	OptionalSignalPresence down_port_;
	SignalPresence up_port_;
	SignalPresence down_port_;
};

class GarageCounterMorphism
{
	public:

		class InternalHom
		{
			public:

				InternalHom(const InputPorts& input_ports);

				std::pair<unsigned int, unsigned int> operator()(const unsigned int state);

				// Accessors

				InputPorts inputs() const
				{
					return inputs_;
				}

			private:

				InputPorts inputs_;

		};

		InternalHom operator()(const InputPorts& input_ports);
};

// Old, draft, version.
class GarageCounterMorphism1
{
	public:

		//using OptionalCount = std::optional<unsigned int>;

		GarageCounterMorphism1(const InputPorts& input_ports);

		std::pair<unsigned int, unsigned int> operator()(const unsigned int state);

		// Accessors

		InputPorts inputs() const
		{
			return inputs_;
		}

	private:

		InputPorts inputs_;
};

} // namespace GarageCounter

namespace ModestThermostat
{

enum class State : char
{
	cooling,
	heating
};

enum class Output : char
{
	heatOn,
	heatOff
};

//------------------------------------------------------------------------------
/// \class InternalHom
/// \ref pp. 51, 52 3.3 Finite-State Machines, Lee & Seshia, Introduction to
/// Embedded Systems
//------------------------------------------------------------------------------
class InternalHom
{
	public:

		InternalHom(
			const double heat_on_limit,
			const double heat_off_limit,
			const double temperature);

		// Output is optional because outputs will only be present only when a
		// change in status of the heater is needed (i.e., when it's on and needs to
		// be turned off, or when it's off and needs to be turned on).
		using OptionalOutput = std::optional<Output>;

		std::pair<State, OptionalOutput> operator()(const State state);

	private:

		const double heat_on_maximum_temperature_;
		const double heat_off_minimum_temperature_;
		double temperature_;
};

class ModestThermostatMorphism
{
	public:

		ModestThermostatMorphism(
			const double heat_on_limit,
			const double heat_off_limit);

		InternalHom operator()(const double temperature);

	private:

		// temperature <= 18 / heatOn
		double heat_on_maximum_temperature_;

		// temperature >= 22 / heatOff
		double heat_off_minimum_temperature_;
};

} // namespace ModestThermostat

// This is an example of a time-triggered finite state machine.
// It keeps track of the passage of time.

namespace TrafficLight
{

// Input.
enum class Pedestrian : char
{
	absent,
	present
};

enum class State : char
{
	red,
	green,
	pending,
	yellow
};

// Output.
enum class Signal : char
{
	signal_red,
	signal_green,
	signal_yellow
};

// Create Output struct for OptionalSignal, Optional count reset
struct Output
{
	using OptionalSignal = std::optional<Signal>;

	OptionalSignal signal_;
	bool reset_count_;
};

class InternalHom
{
	public:

		InternalHom(
			const unsigned int count,
			const Pedestrian pedestrian,
			const unsigned int count_time,
			const unsigned int yellow_to_red_time);

		std::pair<State, Output> operator()(const State state);

	private:

		unsigned count_;
		Pedestrian pedestrian_;
		const unsigned int count_time_;
		const unsigned int yellow_to_red_time_;
};


//------------------------------------------------------------------------------
/// \class InternalHom
/// \ref pp. 62, Figure 3.10, Lee & Seshia, Introduction to Embedded Systems
/// (2016)
//------------------------------------------------------------------------------

class TrafficLightMorphism
{
	public:

		TrafficLightMorphism(
			const unsigned int count_time,
			const unsigned int yellow_to_red_time);

		InternalHom operator()(
			const unsigned int count,
			const Pedestrian pedestrian);

	private:

		unsigned int count_time_;
		unsigned int yellow_to_red_time_;
};

} // namespace TrafficLight

} // namespace Examples

} // namespace StateMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_STATE_MONAD_EXAMPLES_H