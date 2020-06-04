//------------------------------------------------------------------------------
/// \file MoreMax.h
/// \author 
/// \brief 
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
///-----------------------------------------------------------------------------
#ifndef CPP_TEMPLATES_FUNCTION_TEMPLATES_MORE_MAX_H
#define CPP_TEMPLATES_FUNCTION_TEMPLATES_MORE_MAX_H

#include <type_traits> // std::decay

namespace Cpp
{
namespace Templates
{
namespace FunctionTemplates
{

namespace max1
{

/// \ref VJG (2017), pp. 3, basics/max1.hpp
/// \brief This template def. specifies a family of functions that return the
/// maximum of 2 values.
// The type of these parameters is left open as template parameters.

template <typename T>
T max(T a, T b)
{
	// if b < a then yield a else yield b
	return b < a ? a : b;
	// Values of type T, since C++17, can pass temporaries (rvalues, see Appendix
	// B of VJG (2017)), even if neither copy nor move ctor is valid.
}

} // namespace max1

namespace max_with_const_refs
{

template <typename T>
T max (T const& a, T const& b)
{
	return b < a ? a : b;
}

} // namespace max_with_const_refs

namespace max_auto
{

// cf. 1.3.2 Deducing the Return Type, pp. 11, VJG.
// If a return type depends on template parameters, simplest and best approach
// to deduce return type is to let compiler find out.
template <typename T1, typename T2>
auto max(T1 a, T2 b)
{
	return b < a ? a : b;
}

} // namespace max_auto

namespace max_decltype
{

// Before C++14, in C++11, compiler determine type only by making implementation
// of function part of its declaration

template <typename T1, typename T2>
auto max(T1 a, T2 b) -> decltype(b < a ? a : b)
{
	return b < a ? a : b;
}

} // namespace max_decltype

// However, in any case, definition above has significant drawback:
// It might happen that return type is a reference type, because under some
// conditions T might be a reference. 
// For this reason, you should return the type decayed from T:

namespace max_decltype_decay
{



} // namespace max_decltype_decay

} // namespace FunctionTemplates
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_FUNCTION_TEMPLATES_MORE_MAX_H