#include "Cpp/Utilities/SignalHandler.h"

#include <boost/test/unit_test.hpp>

using Cpp::Utilities::create_signal_handler;
using Cpp::Utilities::InterruptSignalHandler;
using Cpp::Utilities::SignalStatus;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(SignalHandler_tests)
BOOST_AUTO_TEST_SUITE(InterruptSignalHandler_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	SignalStatus status;

	auto signal_handler = create_signal_handler(status);

	//InterruptSignalHandler interrupt_signal_handler {signal_handler};
}

BOOST_AUTO_TEST_SUITE_END() // InterruptSignalHandler_tests
BOOST_AUTO_TEST_SUITE_END() // SignalHandler_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp