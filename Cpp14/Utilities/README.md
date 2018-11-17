[Date and time utilities](https://en.cppreference.com/w/cpp/chrono)

* **Duration** - a duration consists of a span of time, defined as some number of ticks of some time unit; e.g. "42 seconds" could be represented by duration consisting 42 ticks of a 1-sec. time unit. class template `duration`, in `<chrono>`, `std::chrono` namespace.

cf. [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration)
```
<chrono>  

template <
  class Rep,
  class Period = std::ratio<1>
> class duration;
```
Consists of count of ticks of type `Rep` and tick period, where tick period is a compile-time rational constant representing number of seconds from one tick to the next.

Only data stored in `duration` is a tick count of type `Rep`. If `Rep` is floating point, then `duration` can represent fractions of ticks. `Period` included as part of duration's type, and is only used when converting between different durations.

## `std::chrono::duration` Member types

| **Member type** | **Definition** |
| :-------------- | :------------- | 
| `rep`           | Rep, an arithmetic type representing the number of ticks | 
| `period` | `Period` a `std::ratio` representing tick period (i.e. number of seconds per tick) |


# `::close`

cf. [`close(2)` Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/close.2.html)

`close` - close a fd

```
#include <unistd.h>

int close(int fd);
```

## ERRORS for `::close`, `errno`

`EBADF` - `fd` isn't a valid open fd
`EINTR` - `close()` call interrupted by signal. see `signal`
`EIO` - I/O error occurred.
`ENOSPEC`, `EDQUOT`  

# `clock` 

[`clock`, Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/clock.3.html)

clock - determine processor time;

```
#include <time.h>

clock_t clock(void);
```
`clock()` function returns approximation of processor time used by program.

## Return value of `clock`

Value returned is CPU time used so far as a `clock_t`;
  * to get number of seconds used, divide by `CLOCKS_PER_SEC`.
If processor time used isn't available or its value can't be represented, function returns value (`clock_t`) -1

## `clock` Attributes

| Interface | Attribute | Value |
| :-------- | --------- | ----- |
| `clock()` | Thread safety | MT-Safe |


