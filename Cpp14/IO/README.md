# `IO` - IO

## `epoll`

cf. https://linux.die.net/man/4/epoll

epoll - IO event notification facility.

3 system calls provided to set up and control `epoll` set:
`epoll_create`, `epoll_ctl`, `epoll_wait`

**epoll** set connected to fd created by `epoll_create`. 
Interest for certain fds registered via `epoll_ctl`.
Finally, actual wait is started by `epoll_wait`



**epoll** event distribution interface able to behave both as Edge Triggered (ET) and Level Triggered (LT). 

Difference between ET and LT event distribution mechanism:
Suppose:
1. fd that represents read side of a pipe (**RFD**) added inside **epoll** device
2. Pip writer writes 2Kb of data on write side of pipe.
3. Call to `epoll_wait` done that'll return **RFD** as ready fd
4. Pipe reader reads 1Kb of data from **RFD**
5. Call to `epoll_wait` done

If **RFD** fd added to **epoll** interface using `EPOLLET` flag, 
call to `epoll_wait` done in 5. will probably hang because available data still present in file input buffers, and remote peer might expecting response based on data it already sent.
  - reason for this is ET event distribution delivers events only when events happens on monitored file.
  - so, in 5., caller might end up waiting for some data that's already present inside input buffer.

In above example, event on **RFD** will be generated because of write done in 2., and event is consumed in 3.
Since read operation done in 4. doesn't consume whole buffer data, call to `epoll_wait` done in 5. might lock indefinitely.
**epoll** interface, when used with `EPOLLET` flag (ET) should use non-blocking fds to avoid having blocking read or write starve task that's handling multiple fds.
Suggested way to use **epoll** as ET (`EPOLLET`) interface is below, possible pitfalls to avoid follow.
  i. with non-blocking fds
  ii. by going to wait for event only after `read`, or `write`
  return `EAGAIN`

On contrary, when used as LT, **epoll** is by all means a faster `poll`, and can be used wherever the latter is used since it shares the same semantics. 
Since even with ET `epoll`, multiple events can be generated up on receiving multiple chunks of data, 
caller has option to specify `EPOLLONESHOT` flag, to tell **epoll** to disable associated fd, after receiving event with `epoll_wait`. 
  - When `EPOLLONESHOT` flag specified, it's caller's responsibility to rearm fd using `epoll_ctl` with `EPOLL_CTL_MOD`

### An Example for Suggested Usage (`epoll`)

### `epoll_create`

cf. [epoll_create, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/epoll_create.2.html)

```
#include <sys/epoll.h>

int epoll_create(int size);
int epoll_create1(int flags);
```

`epoll_create()` creates new `epoll` instance. Since Linux 2.6.8, `size` argument ignored, but must be greater than 0.

`epoll_create` returns fd referring to new epoll instance.
  * This fd used for all subsequent calls to `epoll` interface.
  * when no longer required, fd returned by `epoll_create()` **should be closed** by using `close`.
  * when all fds referring to epoll instance have been closed, kernel destroys instance and releases associated resources for reuse.

`epoll_create1()` - if `flags` is 0, then, other than fact that obsolete `size` argument dropped, `epoll_create1()` same as `epoll_create()`. 
  * following value can be included in `flags` to obtain different behavior:
    - `EPOLL_CLOEXEC` - set close-on-exec (`FD_CLOEXEC`) flag on new fd. See description of `O_CLOEXEC` flag in `open` for reasons why this maybe useful.

#### ERRORS for `epoll_create`, `errno`s

`EINVAL` - `size` isn't positive
`EINVAL` - (`epoll_create1()`) Invalid value specified in `flags`
`EMFILE` - per-user limit on number of epoll instances imposed by `/proc/sys/fs/epoll/max_user_instances` encountered.
`EMFILE` - per-process limit on number of open fds reached
`ENFILE` system-wide limit on total number of open files reached
`ENOMEM` insufficient memory to create kernel object.

### `epoll_ctl`

cf. [epoll_ctl (Linux Programmer's Manual)](http://man7.org/linux/man-pages/man2/epoll_ctl.2.html)

`epoll_ctl` - control interface for an epoll file descriptor

```
#include <sys/epoll.h>

int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
```

Performs control operations on `epoll` instance, referred to by fd `epfd`. It requests that operation `op` be performed for the target fd, `fd`.

Valid values for `op` are 
- `EPOLL_CTL_ADD` - register target fd `fd` on **epoll** instance, referred to by fd `epfd`, and associate event `event` with interal file linked to `fd`
- `EPOLL_CTL_MOD` - change event `event` associated with target fd `fd`
- `EPOLL_CTL_DEL` - remove (degister) target `fd` from **epoll** instance referred to by `epfd`; `event` is ignored, and can be NULL

`event` argument describes object linked to fd `fd`.
`struct epoll_event` defined as:

```
typedef union epoll_data 
{
  void *ptr;
  int fd;
  uint32_t u32;
  uint64_t u64;
} epoll_data_t;

struct epoll_event
{
  uint32_t events; // Epoll events
  epoll_data_t data; // User data variable
}
```

#### Errors of `epoll_ctl`, `errno`s

- `EBADF` - `epfd` or `fd` not a valid fd.


### `epoll_wait`

cf. [`epoll_wait` (Linux Programmer's Manual)](http://man7.org/linux/man-pages/man2/epoll_wait.2.html)

`epoll_wait`, `epoll_pwait` - wait for an I/O event on an epoll fd

```
#include <sys/epoll.h>

int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);

int epoll_pwait(int epfd, struct epoll_event *events,
  int maxevents, int timeout, const sigset_t *sigmask);
```

`epoll_wait()` waits for events on the `epoll` instance referred to by fd `epfd`. 
Memory area pointed to by `events` will contain the events that'll be available for the caller.
Up to `maxevents` are returned by `epoll_wait()`. `maxevents` must be greater than 0.

`timeout` specifies number of **milliseconds** that `epoll_wait()` will block. Time measured against `CLOCK_MONOTONIC` clock. Call will block until either:
* a fd delivers an event
* call is interrupted by a signal handler; or 
* timeout expires.

Note that `timeout` interval will be rounded up to system clock granularity, and kernel scheduling delays mean that blocking interval may overrun by small amount.
Specifying `timeout` of -1 causes `epoll_wait()` to block indefinitely, while specifying `timeout` equal to 0 cause `epoll_wait()` to return immediately, even if no events are available.

#### Return Value of `epoll_wait`

When successful, `epoll_wait()` returns the *number of fds ready for the requested I/O*, or 0 if no fd became ready during requested `timeout` milliseconds.


## `eventfd` - create a fd for event notification

cf. [`eventfd(2)`, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/eventfd.2.html), https://linux.die.net/man/2/eventfd

```
#include <sys/eventfd.h>

int eventfd(unsigned int initval, int flags);
```

`eventfd()` creates 


