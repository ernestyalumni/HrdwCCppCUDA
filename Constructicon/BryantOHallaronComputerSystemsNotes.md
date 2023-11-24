# Notes on Bryant and O'Hallaron Computer Systems: A Programmer's Perspective

Randal E. Bryant. David R. O'Hallaron. **Computer Systems: A Programmer's Perspective.** Third Edition. Pearson.

## Ch. 12. Concurrent Programming

*concurrent programs* - applications that use application-level concurrency
Modern operating systems provide 3 basic approaches for building concurrent programs:
*Processes* - each logical control flow is a process that's scheduled and maintained by the kernel. Since processes have separate virtual address spaces, flows that want to communicate with each other must use some kind of explicit *interprocess communication* (IPC) mechanism.

*I/O multiplexing*. applications explicitly schedule their own logical flows in the context of a single process. Logical flows are modeled as state machines that main program explicitly transitions from state to state as a result of data arriving on file descriptors.
- Since program is a single process, all flows share same address space.

*Threads* Threads are logical flows that run in the context of a single process and are scheduled by kernel. Threads share the same virtual address space like I/O multiplexing flows.

### 12.1.2 Pros and Cons of Processes

Processes file tables are shared between parents and children, and user address spaces are not.

Pro: It's impossible for 1 process to accidentally overwrite virtual memory of another process.
Con: To share information, they must use explicit IPC (interprocess communications)


