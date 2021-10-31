#ifndef UTILITIES_TIME_TIME_SPECIFICATION
#define UTILITIES_TIME_TIME_SPECIFICATION

#include <cstddef> // std::size_t
#include <ctime>
#include <ostream> // std::ostream

namespace Utilities
{

namespace Time
{

//------------------------------------------------------------------------------
/// \brief ::timespec struct wrapper using composition.
///
/// struct timespec
/// {
///   time_t tv_sec; // Seconds
///   long tv_nsec; // Nanoseconds
/// };
//------------------------------------------------------------------------------
class TimeSpecification : public ::timespec
{
  public:

    TimeSpecification();

    explicit TimeSpecification(const ::timespec& timespec);

    std::size_t size_of_tv_sec() const
    {
      return sizeof(timespec_.tv_sec);
    }

    std::size_t size_of_tv_nsec() const
    {
      return sizeof(timespec_.tv_nsec);
    }

    // Queries alignment requirements in bytes.

    static std::size_t alignment_of_tv_sec()
    {
      return alignof(::time_t);
    }

    static std::size_t alignment_of_tv_nsec()
    {
      return alignof(long);
    }

    static std::size_t alignment_of_this()
    {
      return alignof(TimeSpecification);
    }

    ::timespec get_timespec() const
    {
      return timespec_;
    }

    ::timespec* to_timespec_pointer() const
    {
      return &timespec_;
    }

    ::timespec* to_timespec_pointer()
    {
      return &timespec_;
    }

    ::timespec* as_timespec_pointer() const
    {
      return reinterpret_cast<const ::timespec*>(this);
    }

    ::timespec* as_timespec_pointer()
    {
      return reinterpret_cast<::timespec*>(this);
    }

    friend std::ostream& operator<<(
      std::ostream& os,
      const TimeSpecification& time_specification);

  private:

    ::timespec timespec_;
};

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_TIME_SPECIFICATION