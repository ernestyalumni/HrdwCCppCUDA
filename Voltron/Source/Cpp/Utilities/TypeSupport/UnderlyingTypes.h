//------------------------------------------------------------------------------
/// \file UnderlyingTypes.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/types/underlying_type
//------------------------------------------------------------------------------

#ifndef CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H
#define CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H

#include <type_traits> // std::underlying_type
#include <utility>

namespace Cpp
{
namespace Utilities
{

namespace TypeSupport
{

template <typename Enumeration>
auto get_underlying_value(Enumeration&& enum_value)
{
  return 
    static_cast<std::underlying_type_t<Enumeration>>(
      std::forward<Enumeration>(enum_value));
}

template <typename Enumeration>
auto get_underlying_value(Enumeration& enum_value)
{
  return 
    static_cast<std::underlying_type_t<Enumeration>>(
      std::forward<Enumeration>(enum_value));
}

} // namespace TypeSupport
} // namespace Utilities
} // namespace Cpp

#endif // CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H