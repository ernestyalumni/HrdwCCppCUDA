[Date and time utilities](https://en.cppreference.com/w/cpp/chrono)

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


