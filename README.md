# HrdwCCppCUDA
Hardware C, C++, and CUDA; code demonstrating C, C++, CUDA interactions with hardware

- `./learn-c-the-hard-way-lectures`, from Zed Shaw (original author)'s github repository, [`learn-c-the-hard-way-lectures`](https://github.com/zedshaw/learn-c-the-hard-way-lectures)

# `gdb`  

I had already found `gdb` installed on Fedora 25 Workstation Linux.  Note that Mac OS X uses `lldb`. 

However, `gdb` did prompt me to install "Missing separate debuginfos."  In SuperUser, I did this:

```  
sudo dnf debuginfo-install glibc-2.24-10.fc25.x86_64  
```   
which installed  
```  
  glibc-debuginfo.x86_64 2.24-10.fc25                                           
  glibc-debuginfo-common.x86_64 2.24-10.fc25     
```

```
gdb --batch --ex run --ex bt --ex q --args ./ex3
```  
This command tells it to run all the options.  

### `gdb` variables

The "weird" `$NN` in front, at first, are `gdb` variables, that you can access for convenience, later.  

[8 gdb tricks you should know, ksplice blog on Oracle](https://blogs.oracle.com/ksplice/8-gdb-tricks-you-should-know)

## `gdb` commands (API guide)

Compilation of `gdb` commands I use  

### `break`  

[`break` *location*](https://sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html), where, according to the `gdb` documentation, *location*, can be specified in [3 ways](): [Linespec](https://sourceware.org/gdb/onlinedocs/gdb/Linespec-Locations.html#Linespec-Locations), [Explicit](https://sourceware.org/gdb/onlinedocs/gdb/Explicit-Locations.html#Explicit-Locations), [Address](https://sourceware.org/gdb/onlinedocs/gdb/Address-Locations.html#Address-Locations).    

`break` on a line: 

 The command to set a breakpoint is break. If you only have one source file, you can set a breakpoint like so:
```  
    (gdb) break 19
    Breakpoint 1 at 0x80483f8: file test.c, line 19
```  
If you have more than one file, you must give the break command a filename as well:
```  
    (gdb) break test.c:19
    Breakpoint 2 at 0x80483f8: file test.c, line 19  
```  

cf. [4. How do I use breakpoints? ](http://www.unknownroad.com/rtfm/gdbtut/gdbbreak.html)

### `i` , `i[nfo]`

`i[nfo] b` -	[List breakpoints](https://ccrma.stanford.edu/~jos/stkintro/Useful_commands_gdb.html)

### 'info frame'  

`info frame` to show the stack frame info

cf. [how can one see content of stack with gdb](https://stackoverflow.com/questions/7848771/how-can-one-see-content-of-stack-with-gdb)

### `kill`  
Kill the program being debugged? (y or n) y


### `p`  

Finding (memory) address of a variable, e.g.  
```  
p &arr
``` 


cf. [Finding address of a local variable in C with GDB](https://stackoverflow.com/questions/10835822/finding-address-of-a-local-variable-in-c-with-gdb)

## Possibly useful links about `gdb`  

* [disassemble command; Disassembles a specified function or a function fragment.](http://visualgdb.com/gdbreference/commands/disassemble)  
- [gdb – Assembly Language Debugging 101 from mohit.io](http://mohit.io/blog/gdb-assembly-language-debugging-101/)
- [GDB cheatsheet](http://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)
