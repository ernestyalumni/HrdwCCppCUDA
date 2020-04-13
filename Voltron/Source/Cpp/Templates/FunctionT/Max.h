//------------------------------------------------------------------------------
/// \file Max.h
/// \author 
/// \brief 
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
///-----------------------------------------------------------------------------
#ifndef CPP_TEMPLATES_FUNCTION_TEMPLATES_MAX_H
#define CPP_TEMPLATES_FUNCTION_TEMPLATES_MAX_H

namespace Cpp
{
namespace Templates
{
namespace FunctionTemplates
{

// Values of type T, since C++17, can pass temporaries (rvalues, see Appendix B
// of VJG (2017)), even if neither copy nor move ctor is valid.

template <typename T>
T max(T a, T b)
{
	// If b < a then yield a, else yield b
	return b < a ? a : b;
}

template <typename T>
T max_with_const(T const& a, T const& b)
{
  return b < a ? a : b;
}

template <typename T1, typename T2, typename RT>
RT max (T1 a, T2 b);

// When there's no connection between template and call parameters and when
// template parameters can't be determined, must specify template argument
// explicitly with the call.
// However, template argument deduction doesn't take return types into account.
// In C++, return type also can't be deduced from context in which caller uses
// the call.
template <typename T1, typename T2, typename RT>
RT max (T1 a, T2 b)
{
  return b < a ? a : b;
}

template <typename RT, typename T1, typename T2>
RT max (T1 a, T2 b);

template <typename RT, typename T1, typename T2>
RT max (T1 a, T2 b)
{
  return b < a ? a : b;
}

// cf. basics/max2.cpp
// VJG, pp. 15, Sec. 1.5. Overloading Function Templates
// A nontemplate function can coexist with a function template that has the
// same name, and can be instantiated with same type. All other factors being
// equal, overload resolution process prefers nontemplate over 1 generated from
// template.
int max(int a, int b)
{
  return b < a ? a : b;
}


} // namespace FunctionTemplates
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_FUNCTION_TEMPLATES_MAX_H