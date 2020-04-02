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

} // namespace FunctionTemplates
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_FUNCTION_TEMPLATES_MAX_H