#ifndef UTILITIES_TIME_CHRONO_H
#define UTILITIES_TIME_CHRONO_H

#include <chrono>
//#include <type_traits>

namespace Utilities
{
namespace Time
{

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/41850985/check-if-a-template-parameter-is-some-type-of-chronoduration/41851068
//------------------------------------------------------------------------------

template <class T>
struct is_duration : std::false_type
{};

template <class Rep, class Period>
struct is_duration<std::chrono::duration<Rep, Period>> : std::true_type
{};

//------------------------------------------------------------------------------
/// \brief Class template specializations that represents a time interval.
/// \details For std::chrono::duration
/// \ref https://en.cppreference.com/w/cpp/chrono/duration
//------------------------------------------------------------------------------
using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

//------------------------------------------------------------------------------
/// \brief Function template that converts to ToDuration.
/// \details For std::chrono::duration_cast
/// \ref https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
/// https://stackoverflow.com/questions/26184190/alias-a-templated-function
//------------------------------------------------------------------------------
template <
  class ToDuration,
  class T,
  // assignment = 0 is there for default template parameter to hide it.
  typename std::enable_if_t<is_duration<ToDuration>::value, int> = 0
  >
constexpr ToDuration duration_cast(const T& d)
{
  return std::chrono::duration_cast<ToDuration>(d);
}

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_CHRONO_H
