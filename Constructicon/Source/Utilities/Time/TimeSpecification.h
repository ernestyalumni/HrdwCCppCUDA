#ifndef UTILITIES_TIME_TIME_SPECIFICATION_H
#define UTILITIES_TIME_TIME_SPECIFICATION_H

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
class TimeSpecification
{
  public:

    TimeSpecification();

    virtual ~TimeSpecification() = default;

    explicit TimeSpecification(
      const long time_value_sec,
      const long time_value_nsec = 0);

    explicit TimeSpecification(const ::timespec& timespec);

    ::timespec get_timespec() const
    {
      return time_specification_;
    }

    ::time_t get_seconds() const
    {
      return time_specification_.tv_sec;
    }

    long get_nanoseconds() const
    {
      return time_specification_.tv_nsec;
    }

    //--------------------------------------------------------------------------
    /// \details You DO NOT want to reinterpret_cast this object as a ::timespec
    /// because the first 8 bytes are for the overhead of having a class with
    /// composition, as opposed to inheriting directly. This can be seen
    /// explicitly by doing the following:
    ///
    /// ::timespec* as_timespec_pointer()
    /// { return reinterpret_cast<::timespec*>(this)};
    ///
    /// and passing the resulting ::timespec pointer into a system call such as
    /// ::clock_gettime() - it results in the tv_sec field being in tv_nsec
    /// field. It's because the first few (8) bytes is for the class object
    /// pointer.
    //--------------------------------------------------------------------------

    const ::timespec* to_timespec_pointer() const
    {
      return &time_specification_;
    }

    ::timespec* to_timespec_pointer()
    {
      return &time_specification_;
    }

    bool operator>=(const TimeSpecification& rhs) const;

    TimeSpecification operator-(const TimeSpecification& rhs) const;

    friend std::ostream& operator<<(
      std::ostream& os,
      const TimeSpecification& time_specification);

  protected:

    std::size_t size_of_tv_sec() const
    {
      return sizeof(time_specification_.tv_sec);
    }

    std::size_t size_of_tv_nsec() const
    {
      return sizeof(time_specification_.tv_nsec);
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

  private:

    ::timespec time_specification_;
};

namespace Details
{

::timespec carry_or_borrow_nanoseconds(const ::timespec& time_specification);

TimeSpecification carry_or_borrow_nanoseconds(
  const TimeSpecification& time_specification);

} // namespace Details

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_TIME_SPECIFICATION_H