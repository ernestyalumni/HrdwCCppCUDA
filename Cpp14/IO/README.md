# `IO` - IO

## `epoll`



## `eventfd` - create a fd for event notification

cf. [`eventfd(2)`, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/eventfd.2.html), https://linux.die.net/man/2/eventfd

```
#include <sys/eventfd.h>

int eventfd(unsigned int initval, int flags);
```

`eventfd()` creates 


