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
