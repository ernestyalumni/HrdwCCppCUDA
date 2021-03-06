//------------------------------------------------------------------------------
// \file ErrorHandling_tests.cpp
//------------------------------------------------------------------------------
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

#include <boost/test/unit_test.hpp>
#include <cmath>

using Cpp::Utilities::TypeSupport::get_underlying_value;
using Utilities::ErrorHandling::ErrorCodeNumber;
using Utilities::ErrorHandling::HandleReturnValuePassively;
using OptionalErrorNumber =
	Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ErrorHandling_tests)

// cf. https://en.cppreference.com/w/cpp/error/errno
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HandleReturnValuePassivelyGetsLatestErrno)
{
  double not_a_number {std::log(-1.0)};

	const OptionalErrorNumber result {HandleReturnValuePassively()(-1)};

	BOOST_TEST(static_cast<bool>(result));
	BOOST_TEST((*result).error_number() ==
		get_underlying_value(ErrorCodeNumber::argument_out_of_domain));
}

BOOST_AUTO_TEST_SUITE_END() // ErrorHandling_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities