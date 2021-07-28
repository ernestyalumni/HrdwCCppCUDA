#ifndef UTILITIES_TIME_SUPER_TIME_SPECIFICATION
#define UTILITIES_TIME_SUPER_TIME_SPECIFICATION

#include <cstddef> // std::size_t
#include <time.h>

namespace Utilities
{

namespace Time
{

class SuperTimeSpecification : public ::timespec
{
  public:

    SuperTimeSpecification() :
      ::timespec{0, 0}
    {}

    std::size_t size_of_tv_sec() const
    {
      return sizeof(this->tv_sec);
    }

    std::size_t size_of_tv_nsec() const
    {
      return sizeof(this->tv_nsec);
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
      return alignof(SuperTimeSpecification);
    }

    ::timespec* as_timespec_pointer()
    {
      return reinterpret_cast<::timespec*>(this);
    }
};

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_SUPER_TIME_SPECIFICATION