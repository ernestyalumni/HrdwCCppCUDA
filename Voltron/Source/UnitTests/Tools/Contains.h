//------------------------------------------------------------------------------
/// \file Contains.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://stackoverflow.com/questions/29542882/boost-unit-testing-and-catching-exceptions
/// \brief Provide predicates for Boost unit testing. In particular, provide
/// them for BOOST_CHECK_EXCEPTION.
//------------------------------------------------------------------------------
#ifndef UNIT_TESTS_TOOLS_CONTAINS_H
#define UNIT_TESTS_TOOLS_CONTAINS_H

#include <boost/algorithm/string.hpp>
#include <string>

namespace UnitTests
{

namespace Tools
{

//auto error_contains(const std::string& substring);


auto error_contains = [](const std::string& substring)
{
	return [&substring](const auto& err) -> bool
	{
		return boost::algorithm::contains(std::string{err.what()}, substring);
	};
};

} // namespace Tools
} // namespace UnitTests

#endif // UNIT_TESTS_TOOLS_CONTAINS_H
