# `::fork`

cf. [FORK(2) Linux Programmer's Manual](http://man7.org/linux/man-pages/man2/fork.2.html)

```
#include <sys/types.h>
#include <unistd.h>

pid_t fork(void);
```

`fork()` creates new process by duplicating the calling process.
New process is referred to as the *child* process. The calling process is referred to as the *parent* process.

Child process and parent process run in separate memory spaces. 
- At time of `fork()` both memory spaces have same content. 
  * Memory writes, file mappings (`mmap`), and unmappings (`munmap`) performed by 1 of processes don't affect the other.

