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
#include <exception>
#include <string>

namespace UnitTests
{

namespace Tools
{

//auto error_contains(const std::string& substring);


inline auto error_contains = [](const std::string& substring)
{
	return [&substring](const auto& err) -> bool
	{
		return boost::algorithm::contains(std::string{err.what()}, substring);
	};
};

//------------------------------------------------------------------------------
/// \brief Function object to determine if substring is contained in a given
/// string.
//------------------------------------------------------------------------------
class Contains
{
  public:

    explicit Contains(const std::string& given_string);

    bool operator()(const std::string& input_string);

    bool operator()(const std::exception& e);

    // Move constructor and move assignment default to delete because the class
    // data member is const.

  private:

    const std::string given_string_;
};

} // namespace Tools
} // namespace UnitTests

#endif // UNIT_TESTS_TOOLS_CONTAINS_H
