#ifndef UTILITIES_DEBUG_H
#define UTILITIES_DEBUG_H

#include <iostream>
#include <iomanip> // std::setprecision

namespace Utilities
{

//------------------------------------------------------------------------------
/// \ref 5.2 Utility Functions, Implementing Useful Algorithms in C++. Kedyk.
/// \url https://github.com/dkedyk/ImplementingUsefulAlgorithms/blob/master/Utils/Debug.h
//------------------------------------------------------------------------------

// Print the expression, a space, and a new line.
#define DEBUG(variable) std::cout << #variable " " << std::setprecision(17) << (variable) << std::endl;

} // namespace Utilities

#endif // UTILITIES_ENDIAN_CONVERSIONS_H