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
