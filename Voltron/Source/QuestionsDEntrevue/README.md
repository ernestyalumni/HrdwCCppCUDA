[Udacity, How to solve problems](https://youtu.be/UyY0NjtGu7g)

What is the first thing we should do to solve a problem like this one?

- Make sure we understand the problem.

# Rough Notes on Page Faults

See the `HrdwCCppCUDA_grande.pdf` for the polished notes.

https://scoutapm.com/blog/understanding-page-faults-and-memory-swap-in-outs-when-should-you-worry

## About pages

Linux allocates memory to processes by dividing physical memory into pages, and then mapping those physical pages to virtual memory needed by a process. It does this in conjunction with the Memory Management Unit (MMU) in the CPU.
- Typically a page will represent 4KB of physical memory.
- Statistics and flags are kept about each page to tell Linux the status of that chunk of memory.

## Page Fault Example
