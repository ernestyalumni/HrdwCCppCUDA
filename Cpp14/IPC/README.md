# Inter-Process Communication (IPC)

## Message Queue

Message Queues are similar to pipes in that they're opened and closed and have readers and writers.

### `mqueue.h`, `mqd_t`

[`<mqueue.h>`](http://pubs.opengroup.org/onlinepubs/7908799/xsh/mqueue.h.html)

[`<mqueue.h>`](http://pubs.opengroup.org/onlinepubs/009696799/basedefs/mqueue.h.html)

```
#include <mqueue.h>
```
#### Types in `<mqueue.h>`
* **`mqd_t`** - used for message queue descriptors. This will not be an array type. May be implemented using a file descriptor.
* **`mq_attr`** - struct, used in getting and setting attributes of a message queue

##### `mq_attr`
```
long mq_flags // Message queue flags.
long mq_maxmsg // Maximum number of messages.
long mq_msgsize // Maximum message size.
long mq_curmsgs // Number of messages currently queued.
```

##### `mq_getattr`, `mq_setattr` - get/set message queue attributes

[`mq_getattr(3)` - Linux man page](https://linux.die.net/man/3/mq_getattr)


```
#include <mqueue.h>

int mq_getattr(mqd_t mqdes, struct mq_attr *attr);

int mq_setattr(mqd_t mqdes, struct mq_attr *newattr,
  struct mq_attr *oldattr);
```
Link with `-lrt`

`mq_getattr()`, `mq_setattr()` respectively retrieve and modify attributes of the message queue referred to by the descriptor `mqdes`. 

`mq_getattr()` returns `mq_attr` structure in buffer pointed by `attr`:

```
struct mq_attr
{
  long mq_flags;  // Flags: 0 or O_NONBLOCK 
  long mq_maxmsg; // Max. # of messages on queue
  long mq_msgsize;  // Max. message size (bytes)
  long mq_curmsgs;  // # of messages currently in queue
}
```
`mq_setattr()` sets message queue attributes using information supplied in the `mq_attr` structure pointed to by **`newattr`**. The only attribute that can be modified is setting of `O_NONBLOCK` flag in `mq_flags`. Other fields in `newattr` are ignored. 

If `oldattr` field isn't NULL, then buffer that it points to is used to return an `mq_attr` structure that contains same information that's returned by `mq_getattr()`. 

On success `mq_getattr()` and `mq_setattr()` **return** 0; on error, -1 is **returned**, with `errno` set to indicate error.

### Create a Message Queue; `mq_open`

cf. [MQ_OPEN(3) Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/mq_open.3.html)


```
#include <fcntl.h>  // For O_* constants
#include <sys/stat.h>   // For mode constants
#include <mqueue.h>

mqd_t mq_open(const char* name, int oflag);
mqd_t mq_open(const char* name, int oflag, mode_t mode,
  struct mq_attr *attr);
```

`mq_open` - open a message queue; creates a new POSIX message queue,  
  or opens an existing queue.

queue is identified by *name*. 

**`oflag`** - operation flags - specifies flags that control operation of call (definitions of flags values obtained by including `<fcntl.h>`). Exactly 1 of following must be specified in `oflag`: 

* `O_RDONLY`
* `O_WRONLY`
* `O_RDWR`

0 or more of following fails can additionally be `OR`ed in `oflag`:
* `O_CLOEXEC`
* `O_CREAT`; 2 additional arguments must be supplied
* `O_EXCL`
* `O_NONBLOCK` - open queue in nonblocking mode. In circumstances where `mq_receive(3)` and `mq_send(3)` would normally block, these functions instead fail.

`oflag`, operation flags controls whether queue is created or merely accessed;
- defined constants specify operation by a process.

**`mode`** - specifies permissions placed on new queue, as for `open(2)`.
- symbolic definitions of permission bits in `<sys/stat.h>`
- used to set rwx permissions.

`mq_open()` **returns** message queue descriptor for use by other message queue functions; on error `mq_open()` returns `(mqd_t) -1`, with `errno` set to indicate error.

### Functions for sending and receiving for Message Queues

```
mq_send
```
cf. [MQ_SEND(3) Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/mq_send.3.html), [IPC CIT595](https://www.seas.upenn.edu/~cit595/cit595s10/lectures/ipc1.pdf)

```
#include <mqueue.h>

int mq_send(mqd_t mqdes, const char *msg_ptr,
  size_t msg_len, unsigned int msg_prio)

#include <time.h>
#include <mqueue.h>

int mq_timedsend(mqd_t mqdes, const char *msg_ptr,
  size_t msg_len, unsigned int msg_prio,
  const struct timespec *abs_timeout);
```
Feature Test Macro Requirements for glibc (see [`feature_test_macros(7)`](http://man7.org/linux/man-pages/man7/feature_test_macros.7.html)

`mq_send()` - adds message pointer to by `msg_ptr` to message queue referred to by message queue descriptor `mqdes`. 

`msg_len` - specifies length of the message pointed to by `msg_ptr`; this length must be less than or equal to the queue's `mq_msgsize` attribute. 0-length messages are allowed. 

`msg_prio` - nonnegative integer that specifies priority of this message. Messages placed on queue in decreasing order of priority, with newer messages of same priority being placed after older messages with same priority.

If message queue already full (i.e., number of messages on queue equals queue's `mq_maxmsg` attribute), then, by default, `mq_send()` blocks until sufficient space becomes available. 

`mq_timedsend()`~ behaves just like `mq_send()` except that if queue is full, and `O_NONBLOCK` not enabled, then `abs_timeout` points to a structure which specifies how long call will block. 

```
struct timespec 
{
  time_t tv_Sec;  // seconds
  long tv_nsec; // nanoseconds
}
```

On success, `mq_send()`, `mq_timedsend()` **returns** 0; on error, -1 and `errno` set to indicate error.


```
mq_receive
```
cf. [MQ_RECEIVE(3) Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/mq_receive.3.html), [IPC CIT595](https://www.seas.upenn.edu/~cit595/cit595s10/lectures/ipc1.pdf)

```
#include <mqueue.h>

ssize_t mq_receive(mqd_t mqdes, char *msg_ptr,
  size_t msg_len, unsigned int* msg_prio)

#include <time.h>
#include <mqueue.h>

ssize_t mq_timedreceive(mqd_t mqdes, char *msg_ptr,
  size_t msg_len, unsigned int* msg_prio,
  const struct timespec *abs_timeout);
```
Feature Test Macro Requirements for glibc (see [`feature_test_macros(7)`](http://man7.org/linux/man-pages/man7/feature_test_macros.7.html)

`mq_timedreceive():`
  `_POSIX_C_SOURCE >= 20112L`

`mq_receive()` - removes the oldest message with highest priority adds message pointer to by `msg_ptr` to message queue referred to by message queue descriptor `mqdes`, and places it in buffer pointed to by `msg_ptr`. 

`msg_len` - specifies size of buffer pointed to by `msg_ptr`; this must be greater than or equal to `mq_msgsize` attribute of queue (see `mq_getattr`). 

If `msg_prio` - is not NULL, then buffer to which it points is used to return priority associated with received message.

If message queue empty, then, by default, `mq_receive()` blocks until message becomes available. 

`mq_timedreceive()`~ behaves just like `mq_receive()` except that if queue is empty, and `O_NONBLOCK` not enabled, then `abs_timeout` points to a structure which specifies how long call will block. 

```
struct timespec 
{
  time_t tv_Sec;  // seconds
  long tv_nsec; // nanoseconds
}
```

On success, `mq_receive()`, `mq_timedreceive()` **returns** 0; on error, -1 and `errno` set to indicate error.


#### Functions in `<mqueue.h>` 

```
int mq_close(mqd_t);
int mq_getattr(mqd_t, struct mq_attr*);
int mq_notify(mqd_t, const struct sigevent*);
mqd_t mq_open(const char*, int, ...);

int mq_unlink(const char*);

```

### ** `mq_unlink`** remove a message queue

[MQ_UNLINK, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/mq_unlink.2.html)

`mq_unlink` - remove a message queue

`mq_unlink` - removes the specified message queue name. 
 - message queue name is removed immediately. Queue itself is destroyed once any other processes that have the queue open close their descriptors referring to the queue. 

#### Return Value of `mq_unlink`

On success `mq_unlink` returns 0; on error, -1 is returned, with errno set to indicate the error.

`mq_unlink()` is thread safe, MT-Safe


### **`mq_close`**

[MQ_CLOSE, Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/mq_close.3.html)

`mq_close` - close a message queue descriptor

If calling process has attached a notification request (`mq_notify`) to this message queue via `mqdes`, then this request is removed, and another process can now attach a notification request. 

```
#include <mqueue.h>

int mq_close(mqd_t mqdes);
```
Link with `-lrt`

#### Return Value of `mq_close`

On success, `mq_close()` returns 0, on error -1, returned with `errno` set to indicate the error. 




cf. [OS: IPC I CIT 595 Spring 2010](https://www.seas.upenn.edu/~cit595/cit595s10/lectures/ipc1.pdf)

https://fenix.tecnico.ulisboa.pt/downloadFile/1407993358851967/06-IPC.pdf

https://www.cs.rutgers.edu/%7Epxk/417/notes/content/07-groups-slides.pdf
Message ordering, around pp. 19

https://www.seas.upenn.edu/~cit595/cit595s10/lectures/ipc1.pdf
