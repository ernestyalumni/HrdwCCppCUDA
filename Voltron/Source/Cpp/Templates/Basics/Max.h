//------------------------------------------------------------------------------
/// \file Max.h
/// \author 
/// \brief 
/// \ref Vandevoorde, Josuttis, Gregor. C++ Templates: The Complete Guide. 2nd
/// Ed. Addison-Wesley Professional. 2017.
///-----------------------------------------------------------------------------
#ifndef CPP_TEMPLATES_BASICS_MAX_H
#define CPP_TEMPLATES_BASICS_MAX_H

namespace Cpp
{
namespace Templates
{
namespace Basics
{

template <typename T>
T max(T a, T b)
{
	// If b < a then yield a, else yield b
	return b < a ? a : b;
}

} // namespace Basics
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_BASICS_MAX_H