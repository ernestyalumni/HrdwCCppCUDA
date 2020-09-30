//------------------------------------------------------------------------------
/// \file WrapperPointers.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Pointers under encapsulation.
/// @ref https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/
///-----------------------------------------------------------------------------
#ifndef CPP_UTILITIES_MEMORY_STD_TO_ADDRESS_H
#define CPP_UTILITIES_MEMORY_STD_TO_ADDRESS_H

#include <memory> // std::pointer_traits
#include <type_traits>

namespace Cpp
{
namespace Utilities
{
namespace Memory
{

#if defined __cpp_lib_to_address

using std::to_address;

#else // ! defined __cpp_lib_to_address

// https://en.cppreference.com/w/cpp/memory/to_address

template <class T>
constexpr T* to_address(T* p) noexcept
{
  static_assert(!std::is_function_v<T>);
  return p;
}

/*
template <class T>
constexpr auto to_address(const T& p) noexcept
{
  // C++20 requires  
  //  if constexpr (requires{ std::pointer_traits<T>::to_address(p); })
  {
    return std::pointer_traits<T>::to_address(p);
  }
  else
  {
    return std::to_address(p.operator->());
  }
}
*/

#endif // defined __cpp_lib_to_address

} // namespace Memory
} // namespace Utilities
} // namesapce Cpp

#endif // CPP_UTILITIES_MEMORY_STD_TO_ADDRESS_H