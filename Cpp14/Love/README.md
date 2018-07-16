# Linux System Programming; Systems Programming cf. `Love/`  

cf. [cs 241 spring 2012 Systems Programming; Illinois](https://courses.engr.illinois.edu/cs241/sp2012/)

## Interprocess Communication 

[Interprocess Communication](https://courses.engr.illinois.edu/cs241/sp2012/lectures/29-IPC.pdf)

cf. [Interprocess Communication](https://courses.engr.illinois.edu/cs241/sp2012/lectures/30-IPC.pdf)


#### What if there are multiple producers?

##### Solution (to multiple producers) 
* Need a way to **wait for any 1 of a set of events** to happen
* something similar to `wait()` to wait for any child to finish, but for events on file descriptors. 

### `select` and `poll`

cf. pp. 1326. Ch. 63. Alternative I/O Models. Kerrisk, **The Linux Programming Interface**. 2010. No Starch Press. 

Nonblocking I/O allows us to periodically check ("poll") whether I/O is possible on a file descriptor.

e.g. make an input file descriptor (fd) nonblocking, then periodically perform nonblocking reads. 
If we need to monitor multiple fd's, then mark them all nonblocking, and poll each of them in turn. However, polling in this manner is usually undesirable. If polling done infrequently, latency before application responds to I/O event maybe unacceptably long; on the other hand, polling in a tight loop wastes CPU time.

"poll" - performing a nonblocking check on the status of a fd.

* *I/O multiplexing* allows a process to simultaneously monitor multiple fd's to find out whether I/O is possible on any of them. `select()`, `poll()` perform I/O multiplexing.
* *Signal-driven I/O* - process requests kernel send it a signal when input is available or data can be written on a specified fd. Process can then carry on performing other activities, and is notified when I/O becomes possible via receipt of signal. 
  - When monitoring large numbers of fds, signal-drive I/O provides significantly better performance than `select()` and `poll()`.
* `epoll`, like I/O multiplexing, allows process to monitor multiple fds to see if I/O possible on any of them. Like signal-driven I/O, `epoll` provides much better performance when monitoring large numbers of fds.

In effect, I/O multiplexing, signal-drive I/O, `epoll` are all methods of achieving the same result-monitoring 1 or, commonly, several fd's simultaneously to see if they're *ready* to perform I/O (more precisely, to see whether I/O system call could be performed without blocking).
- transition of fd into ready state triggered by some type of I/O *event*, e.g. arrival of input, completion of socket connection, or availability of space in previously full socket send buffer after TCP transmits queued data to socket peer.

Monitoring multiple fds useful in applications such as network servers that must simultaneously monitor multiple client sockets, or applications that must simultaneously monitor input from terminal and a pipe or socket. 

Note, none of these techniques performs I/O; they merely tell us a file descriptor is ready; other system calls must then be used to actually perform I/O.

POSIX AIO allows process to queue an I/O operation to a file and then later be notified when operation is complete; advantage is that initial I/O call returns immediately, so process is not tied up waiting for data to be transferred to kernel or for operation to complete. Linux provides threads-based implementation of POSIX AIO within *glibc*; cf. Gallmeister, 1995, Robbins & Robbins, 2003.

`epoll` - like signal-drive I/O, allows application to efficiently monitor large numbers of fds; 
  `epoll` provides number of advantages over signal-drive I/O:
  - avoid complexities of dealing with signals
  - specify kind of monitoring that we want to perform (e.g., ready for reading or ready for writing)
  - select either level-triggered or edge-triggered notification

### Level-Triggered and Edge-Triggered Notification

* *level-triggered notification*: fd considered to be ready if it's possible to perform an I/O system call without blocking. 
* *edge-triggered notification*: notification provided if there's I/O activity (e.g. new input) on fd since it was last-monitored.

* *level-triggered notification*
  - can check readinesss of fd any time
  - i.e. we determine fd is ready (e.g., it has input available), can perform some I/O on descriptor, and then repeat monitoring operation to check if descriptor is still ready (e.g. still has more input available), in which case we can perform more I/O, and so on.
  - i.e., because level-triggered model allows us to repeat I/O monitoring operation at any time, it's not necessary to perform as much I/O as possible (e.g., read as many bytes as possible) on fd (or even perform any I/O at all) each time we're notified fd is ready.
* *edge-triggered notification*, receive notification only when I/O event occurs. 
  - don't receive any further notification until another I/O event occurs.
  - when I/O event notified for a fd, usually don't know how much I/O is possible (e.g. how many bytes available for reading). 
  - following rules:
    * after notification of I/O event, program should - at some point - perform as much I/O as possible (e.g., read as many bytes as possible) on corresponding file descriptor. If program fails to do this, then it might miss opportunity to perform some I/O, because it wouldn't be aware of need to operate on fd until another I/O event occurred. This could lead to spurious data loss or blockages. We said "at some point" because sometimes it may not be desirable to perform all I/O immediately after we determine fd ready.
    * if program employs a loop to perform as much I/O as possible on fd, and descriptor marked blocking, then eventually I/O system call will block when no more I/O possible. For this reason, each monitored fd normally placed in nonblocking mode, and after notification of I/O event, I/O operations performed repeatedly until relevant system call (e.g. `read()` or `write()`) fails with error `EAGAIN` or `EWOULDBLOCK`.

*level-triggered notification* - continue to receive events until underlying fd is no longer in a ready state.

*edge-triggered mode* - only receive events when state of watched fd change.

#### `select` and `poll`: Waiting for input

##### Similar parameters (for `select` and `poll`)

* Set of file descriptors (fds)
* Set of events for each descriptor (fd)
* Timeout length

##### Similar return value (for `select` and `poll`)

* Set of file descriptors
* events for each descriptor 

##### Notes (on `select` and `poll`)

* `select` is slightly simpler
* `poll` supports waiting for more event types 
* newer variant available on some systems: `epoll` 

#### `select` 

``` 
#include <sys/time.h> // For portability

int select(int num_fds, fd_set* read_set, 
  fd_set* write_set, fd_set* except_set,
  struct timeval* timeout);
```
* wait for readable/writable file descriptors (fds) 

Returns number of ready file descriptors, 0 on timeout, or -1 on error

##### Returns (`select`)
* number of descriptors ready
* -1 on error, sets `errno`

##### Parameters (of `select`)
* `num_fds` - number of fds to check, numbered from 0
* `read_set`, `write_set`, `except_set` 
  * sets (bit vectors) of file descriptors to check for the specific condition
* `timeout` 
  * time to wait for a descriptor to become ready 

#### File Descriptor Sets

**Bit vectors** - 
  * Often 1024 bits, only first `num_fds` checked 
  * Macros to create and check sets 

``` 
fds_set myset;
void FD_ZERO(&myset);       // clear all bits
void FD_SET(n, &myset);     // set bits n to 1
void FD_CLEAR(n, &mysef);   // clear bit n
int FD_ISSET(n, &myset);    // is bit n set?
``` 

##### 3 conditions to check for (for file descriptor sets)
* Readable 
  * Data available for reading
* Writable
  * Buffer space available for writing
* Exception
  * Out-of-band data available (TCP)


## `epoll`

- central data structure of `epoll` is *epoll instance*, referred to via open fd. This fd isn't used for I/O. Instead, it's a handle for kernel data structures that serve 2 purposes:
  * recording a *interest list*, list of fds that this process has declared an interest in monitoring
  * maintaining list of fds that are ready for I/O - the *ready list*.

For each fd monitored by `epoll`, we can specify a bit mask indicating events that we're interested in knowing about; these bit masks correspond closely to bit masks used with `poll()`. 

`epoll` API consists of 3 system calls:
1. `epoll_create()` creates `epoll` instance and returns a fd referring to the instance.
2. `epoll_ctl()` - manipulates interest list associated with an `epoll` instance; using `epoll_ctrl()`, we can add new fd to list, remove existing descriptor from list, modify mask that determines which events to be monitored for descriptor. 
3. `epoll_wait()` system call returns items from ready list associated with `epoll` instance.

### `epoll_create`, `epoll_create1` 

cf. [`EPOLL_CREATE(2)` Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/epoll_create1.2.html)

``` 
#include <sys/epoll.h>

int epoll_create(int size);
int epoll_create1(int flags);
``` 

`epoll_create()` creates a new `epoll` instance; since Linux 2.6.8, `size` argument is ignored, but must be greater than 0. 

`epoll_create()` returns fd referring to new epoll instance. This fd is used for all subsequent calls to `epoll` interface. 
When no longer required, fd returned by `epoll_create()` should be closed by using `close`. 
When all fds referring to an epoll instance have been closed, kernel destroys instance and releases associated resources for reuse.

`epoll_create1()` - if `flags` is 0, then, other than fact that obsolete `size` argument is dropped, `epoll_create1()` same as `epoll_create()`. 
The following value can be included in `flags`:
  `EPOLL_CLOEXEC` - set close-on-exec (`FD_CLOEXEC`) flag on new fd. 

Return on success, nonnegative fd; on error, -1 returned, `errno` set to indicate error.

*Notes* - Now kernel dynamically sizes required data structures without needing hint `size`, but `size` must still be greater than 0, in order to ensure backwards compatibility.

### `epoll_ctl()`, modifying `epoll` interest list.

[`EPOLL_CTL(2)`, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/epoll_ctl.2.html)

`epoll_ctl` - control interface for an epoll fd.

``` 
#include <sys/epoll.h>

int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
```

Valid values for `op` argument are 
`EPOLL_CTL_ADD`
`EPOLL_CTL_MOD`
`EPOLL_CTL_DEL`

#### Return value of `epoll_ctl`

When **successful**, `epoll_ctrl()` returns **0**. When error occurs, `epoll_ctl()` returns -1.

### `epoll_wait()` 

cf. [`EPOLL_WAIT(2)`](http://man7.org/linux/man-pages/man2/epoll_wait.2.html)

```
#include <sys/epoll.h>

int epoll_wait(int epfd, struct epoll_event *events,
  int maxevents, int timeout);

int epoll_pwait(int epfd, struct epoll_event *events,
  int maxevents, int timeout,
  const sigset_t *sigmask);
```

`epoll_wait()` waits for events on the `epoll` instance, referred to by fd `epfd`. 
Memory area pointed to by `events` will contain events that'll be available for the caller. 
Up to `maxevents` are returned by `epoll_wait()`. `maxevents` must be greater than 0.

`timeout` argument specifies number of milliseconds `epoll_wait()` will block. Time measured against `CLOCK_MONOTONIC` clock. 

The call will block until either: 
* fd delivers an event;
* call is interrupted by a signal handler; or 
* timeout expires

#### Return value of `epoll_wait()`

When successful, `epoll_wait()` returns number of fds ready for the requested I/O, or 0 if no fd became ready during requested `timeout` milliseconds. 

When an error occurs, `epoll_wait()` returns -1 and `errno` is set appropriately.


## Code

[`altio/epoll_input.c`](http://man7.org/tlpi/code/online/dist/altio/epoll_input.c.html), for this repository, in `../IPC/AlternativeIO/`.  

## References

Neil Matthew, Richard Stones. [Beginning Linux Programming](https://doc.lagout.org/operating%20system%20/linux/Beginning%20Linux%20Programming%2C%204%20Ed.pdf). 4th. Ed. 

Michael Kerrisk. [The Linux Programming Interface](https://doc.lagout.org/programmation/unix/The%20Linux%20Programming%20Interface.pdf): A Linux and UNIX System Programming Handbook. 2010. ISBN-13: 978-1-59327-220-3

[This is `lib/tlpi_hrd.h` (Listing 3-1, page 51), an example from the book, *The Linux Programming Interface*](http://man7.org/tlpi/code/online/dist/lib/tlpi_hdr.h.html)

https://suchprogramming.com/epoll-in-3-easy-steps/
