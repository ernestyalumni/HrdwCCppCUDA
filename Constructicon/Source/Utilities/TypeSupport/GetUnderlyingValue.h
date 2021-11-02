//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/types/underlying_type
//------------------------------------------------------------------------------

#ifndef UTILITIES_TYPE_SUPPORT_GET_UNDERLYING_VALUE_H
#define UTILITIES_TYPE_SUPPORT_GET_UNDERLYING_VALUE_H

#include <type_traits> // std::enable_if, std::underlying_type, std::is_enum

namespace Utilities
{
namespace TypeSupport
{

//------------------------------------------------------------------------------
/// \details Recall for std::enable_if,
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

#endif // UTILITIES_TYPE_SUPPORT_GET_UNDERLYING_VALUE_H