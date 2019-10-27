//------------------------------------------------------------------------------
/// \file OnlyOneArgument.h
/// \author Ernest Yeung
/// \brief Type trait for functions with only one argument.
/// \ref https://stackoverflow.com/questions/43526647/decltype-of-function-parameter
///-----------------------------------------------------------------------------
#ifndef _UTILITIES_ONLY_ONE_ARGUMENT_H_
#define _UTILITIES_ONLY_ONE_ARGUMENT_H_

#include <type_traits> // std::enable_if_t, std::is_invocable

namespace Utilities
{

template <typename> struct OnlyOneArgument;

/// \ref https://stackoverflow.com/questions/43526647/decltype-of-function-parameter

template <
  typename ReturnType,
  typename Argument//,
  //std::enable_if_t<std::is_invocable_v<ReturnType(Argument)>, Argument> = 0
	>
struct OnlyOneArgument<ReturnType(Argument)>
{
	//constexpr operator bool() const
	//{
	//	return std::is_invocable_v<ReturnType(Argument), Argument>;
	//}

	static constexpr bool value =
		std::is_invocable_v<ReturnType(Argument), Argument>;
	//static constexpr bool value =
		//std::is_invocable_v<ReturnType(Argument), Argument>;

	//bool operator()()
	//{
		//return value;
	//}

  using type = Argument;
};

template <typename Function>
using OnlyOneArgumentT = typename OnlyOneArgument<Function>::type;


//template <typename Function

} // namespace Utilities

#endif // _UTILITIES_ONLY_ONE_ARGUMENT_H_