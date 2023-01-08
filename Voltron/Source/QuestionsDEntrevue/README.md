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

Virtual Memory -> Page Table (controlled by MMU) -> Physical Memory (RAM)

These pages can be in different states; some free (unused), some used to hold executable code, some allocated as data for a program

## Page Fault Example

Consider a large running program on a Linux system.
- program executable size could be measured in megabytes, but not all that code will run at once.
  * some of the code will only be run during initialization or when a special condition occurs
  * over time Linux can discard pages of memory which hold executable code, if it thinks they're no longer needed or will be used rarely

A program is executed by CPU as it steps its way through the machine code. (EY: so the instructions tell where to go in virtual memory?)
- each instruction is stored in physical memory at a certain address.
  * MMU handles mapping from physical address space to virtual address space.
- At some point program's execution the CPU may need to address code which isn't in memory.
  * The MMU knows that the page for that code isn't available (because Linux told it) and so CPU will raise a **page fault**

Page fault isn't an error, but a known event where the CPU is telling the OS that it needs physical access to more of the code.

### Copy on Write

For data memory used by program, e.g. an executable asks Linux for some memory, e.g. 8 megabytes. Linux doesn't actually give the process 8 megabytes of physical memory. Instead it allocates 8 megabytes of virtual memory and marks those pages as "copy on write,"
* i.e. while they're unused there's no need to actually physically allocate them, but moment the process writes to that page, a real physical page is allocated and the page assigned to the process

o is for user-defined format
```
ps -eo min_flt,maj_flt,cmd
```

## Swapping

Under normal conditions, kernel is managing pages of memory so that virtual address space is mapped onto physical memory and every process has access to data and code it needs.

But what happens when kernel doesn't have any more physical memory left?
- kernel will start to write to disk some of the pages which it's holding in memory, and use newly freed pages to satisfy current page faults.

## tl;dr

* total amount of virtual address space for all running processes far exceeds amount of physical memory
* When CPU needs to access page that isn't in memory, it raises a page fault.
* major page fault is one that can only be satisfied by accessing the disk
* minor page fault satisfied by sharing pages that are already in memory.

