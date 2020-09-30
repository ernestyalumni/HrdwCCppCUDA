//------------------------------------------------------------------------------
/// \file ByteDifference.h
/// \author
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_BYTE_DIFFERENCE_H
#define UTILITIES_BYTE_DIFFERENCE_H

namespace Utilities
{

// Remember, reinterpret_cast is resolved at compile-time; it's nothing more
// than "look to a pointer that's pointing to type A with eyes of who is
// looking for type B".
// https://stackoverflow.com/questions/27309604/do-constant-and-reinterpret-cast-happen-at-compile-time/27309763
template <typename T>
int byte_difference(T* p, T* q)
{
  return reinterpret_cast<char*>(q) - reinterpret_cast<char*>(p);
}

} // namespace Utilities

#endif // UTILITIES_BYTE_DIFFERENCE_H