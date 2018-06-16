# FileIO

cf. Robert Love. **Linux System Programming: Talking Directly to the Kernel and C Library.** Second Edition. O'Reilly Media; Second edition (June 8, 2013). ISBN-10: 1449339530. ISBN-13: 978-1449339531. 

cf. pp. 25 Ch. 2 File I/O. Robert Love. **Linux System Programming: Talking Directly to the Kernel and C Library.** Second Edition. O'Reilly Media; Second edition (June 8, 2013). (2013)

*file table* - kernel maintains a per-process list of open files, table indexed via nonnegative integers known as *file descriptors*  
 - each entry in list contains information about an open file, including 
  * pointer to in-memory copy of file's backing inode and 
  * associated metadata, such as file position and access modes

*file descriptors* - (**often** abbreviated *fds*) nonnegative integers indexing file tables. 
  - both user space and kernel space use file descriptors as unique cookies:
  - opening a file returns a fd, while 
  - subsequent operations (reading, writing, and so on) take fd as their primary argument. 
  - every process by convention has at least 3 fds open:
    * 0, *standard in* (stdin) - normally, stdin connected to terminal's input device (usually user's keyboard) 
    * 1, *standard out* (stdout) - usually connected to terminal's display
    * 2, *standard error* (stderr) - usually connected to terminal's display
    - users can *redirect* these standard fds and even pipe output of 1 program into input of another.


## `open()` system call, Opening Files, 

cf. pp. 26 "The open() System Call". *Opening Files*. Love (2013)

cf. [`OPEN(2)`, Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/open.2.html) 

``` 
int open(const char *pathname, int flags);
int open(const char *pathname, int flags, mode_t mode);
``` 
`open()` system call opens file specified by `pathname`. 

### Flags for `open()` 

Argument `flags` must include 1 of the following *access modes*: `O_RDONLY`, `O_WRONLY` or `O_RDWR`.  
`flags` argument is bitwise-OR of 1 or more flags 
  - `flags` argument may be bitwise-ORed with 0 or more of the following values 


### `creat()` 

``` 
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int creat (const char *name, mode_t mode);

``` 
