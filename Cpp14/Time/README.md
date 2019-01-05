# POSIX timers, `time.h` (time types)

`time.h` is `<ctime>` in C++

[Open Group, UNIX specification, `time.h`](http://pubs.opengroup.org/onlinepubs/7908799/xsh/time.h.html)

`<time.h>` (`<ctime>`) header declares structure **timespec**, which ahs the following members:


# Timer; `timerfd`, `#include <sys/timerfd.h>`

[`timerfd_create` Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/timerfd_create.2.html)

```
int timerfd_create(int clockid, int flags);

int timerfd_settime(int fd, int flags,
  const struct itimerspec *new_value,
  struct itimerspec *old_value);

int timerfd_gettime(int fd, struct itimerspec *curr_value);
```

These system calls create and operate on a timer that delivers timer expiration notifications via a file descriptor.

Starting with Linux 2.6.27, following values may be bitwise ORed in flags to change the behavior of `timerfd_create`:
* `TFD_NONBLOCK`
* `TFD_CLOEXEC` - set the close-on-exec (`FD_CLOEXEC`) flag on the new file descriptor.


**`timerfd_settime()`**
```
int timerfd_settime(int fd, int flags,
  const struct itimerspec *new_value,
  struct itimerspec *old_value);
```

- `new_value` argument specifies initial expiration and interval for the timer.
- `old_value` - if not NULL, then `itimerspec` structure that it points to is used to return the setting of the timer that was current at the time of the call.

- `new_value.it_interval` - set 1 or both fields to nonzero values specifies the period, in seconds and nanoseconds, for repeated timer expirations after initial expiration.
  * If both fields of `new_value.it_interval` are 0, timer expires just once, at time specified by `new_value.it_value`
- `new_value.it_value` - specifies initial expiration of timer, in seconds and nanoseconds.
  * Setting either field of `new_value.it_value` to nonzero value arms timer.
  * Setting both fields of `new_value.it_value` to 0 disarms timer.


```
struct timespec
{
  time_t tv_sec;  // Seconds
  long tv_nsec;   // Nanoseconds
};

struct itimerspec
{
  struct timespec it_interval;  // Interval for periodic timer
  struct timespec it_value;     // Initial expiration.
};
```

An absolute timeout can be selected via the `flags` argument.

The `flags` argument is a bit mask that can include the following values:

- `TFD_TIMER_ABSTIME` - Interpret `new_value.it_value` as an absolute value on the timer's clock. Timer will expire when value of timer's clock reaches value specified in `new_value.it_value`.

The `flags` argument is either 0, to start a relative timer (`new_value.it_value` specifies a time relative to current value of the clock specified by `clockid`), or `TFD_TIMER_ABSTIME`, to start an absolute timer (`new_value.it_value` specifies an absolute time for the clock specified by `clockid`; i.e. timer will expire when value of that clock reaches value specified in `new_value.it_value`.)

cf. https://linux.die.net/man/2/timerfd_settime

