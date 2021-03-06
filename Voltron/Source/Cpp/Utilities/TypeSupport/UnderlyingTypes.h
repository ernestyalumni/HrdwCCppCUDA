//------------------------------------------------------------------------------
/// \file UnderlyingTypes.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/types/underlying_type
//------------------------------------------------------------------------------

#ifndef CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H
#define CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H

#include <type_traits> // std::enable_if, std::underlying_type, std::is_enum
#include <utility>

namespace Cpp
{
namespace Utilities
{

namespace TypeSupport
{

// TODO: Decide when to obsolete this.
/*
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
*/

//------------------------------------------------------------------------------
/// template <bool B, class T = void>
/// struct enable_if;
/// If B is true, std::enable_if has a public member typedef type, equal to T;
/// otherwise, there's no member typedef
//------------------------------------------------------------------------------
template <
  typename Enumeration,
  typename = typename std::enable_if_t<std::is_enum<Enumeration>::value>
  >
constexpr auto get_underlying_value(const Enumeration enum_value) ->
  std::enable_if_t<
    std::is_enum<Enumeration>::value,
    std::underlying_type_t<Enumeration>
    >
{
  return static_cast<std::underlying_type_t<Enumeration>>(enum_value);
}

} // namespace TypeSupport
} // namespace Utilities
} // namespace Cpp

#endif // CPP_UTILITIES_TYPE_SUPPORT_UNDERLYING_TYPES_H