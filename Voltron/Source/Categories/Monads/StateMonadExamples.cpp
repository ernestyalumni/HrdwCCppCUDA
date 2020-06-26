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

GarageCounterMorphism::GarageCounterMorphism(const InputPorts& input_ports):
	inputs_{input_ports}
{}

std::pair<unsigned int, unsigned int> GarageCounterMorphism::operator()(
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

} // namespace Examples

} // namespace StateMonad
} // namespace Monads
} // namespace Categories