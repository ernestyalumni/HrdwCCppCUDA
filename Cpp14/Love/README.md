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
int select(int num_fds, fd_set* read_set, 
  fd_set* write_set, fd_set* except_set,
  struct timeval* timeout);
```
* wait for readable/writable file descriptors (fds) 

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


