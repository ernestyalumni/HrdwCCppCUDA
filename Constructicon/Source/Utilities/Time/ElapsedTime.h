#ifndef UTILITIES_TIME_ELAPSED_TIME_H
#define UTILITIES_TIME_ELAPSED_TIME_H

#include "GetClockTime.h"
#include "TimeSpecification.h"

namespace Utilities
{
namespace Time
{

//------------------------------------------------------------------------------
/// \details Use templates in order to ensconce any virtual table overhead. We
/// want the fastest implementation.
//------------------------------------------------------------------------------  
template <typename GetTimeT>
class ElapsedTime
{
  public:

    ElapsedTime();

    void start()
    {
      start_time_ = get_time_();
    }

    TimeSpecification operator()() const
    {
      return get_time_() - start_time_;
    }

    TimeSpecification get_start_time() const
    {
      return start_time_;
    }

  private:

    GetTimeT get_time_;

    TimeSpecification start_time_;
};

template <typename GetTimeT>
ElapsedTime<GetTimeT>::ElapsedTime():
  get_time_{},
  start_time_{get_time_()}
{}

using MonotonicElapsedTime = ElapsedTime<GetMonotonicClockTime>;
using RealTimeElapsedTime = ElapsedTime<GetRealTimeClockTime>;

} // namespace Time
} // namespace Utilities

#endif // UTILITIES_TIME_ELAPSED_TIME_H