#ifndef UTILITIES_TIME_CHRONO
#define UTILITIES_TIME_CHRONO

#include <chrono>

namespace Utilities
{

namespace Time
{


//------------------------------------------------------------------------------
/// \brief Class template specializations that represents a time interval.
/// \details For std::chrono::duration
/// \ref https://en.cppreference.com/w/cpp/chrono/duration
//------------------------------------------------------------------------------
using Nanoseconds = std::chrono::nanoseconds;
using Seconds = std::chrono::seconds;

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_CHRONO