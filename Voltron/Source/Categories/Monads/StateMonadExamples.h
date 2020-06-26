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

class InternalHom
{
	public:

		InternalHom(const double temperature);

		using OptionalOutput = std::optional<Output>;

		std::pair<State, OptionalOutput> operator()(const State state);

	private:

		double temperature_;
};

class ModestThermostatMorphism
{
	public:

		InternalHom operator()(const double temperature);
};

} // namespace ModestThermostat

} // namespace Examples

} // namespace StateMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_STATE_MONAD_EXAMPLES_H