# HrdwCCppCUDA
Hardware C, C++, and CUDA; code demonstrating C, C++, CUDA interactions with hardware

- `./Cmemory/` - memory layout in C, heap and stack memory management in C, Segmentation Fault cases in C, including stack overflow, buffer overflow, dereferencing pointers to null pointers, wild pointers (uninitialized), dangling pointers, down to instruction level (with `gdb`).  
	* `./Cmemory/heapstack` - heap and stack C examples, including stack overflow, buffer overflow, and (nasty) memory leak (heap overflow) examples.  
	* Reference for *Registers* (x86_64 architecture) are on `./Cmemory/README.md`  

- `./learn-c-the-hard-way-lectures`, from Zed Shaw (original author)'s github repository, [`learn-c-the-hard-way-lectures`](https://github.com/zedshaw/learn-c-the-hard-way-lectures)
- `./Cppcls` - all about C++ classes, C++ class inheritance, `vtable`s, `virtual`    

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

### `c` `continue`

https://www.exploit-db.com/docs/28475.pdf

### `delete [breakpoint]` [delete a break point]() 

### [`disassemble`](http://visualgdb.com/gdbreference/commands/disassemble)  

Disassembles a function or function fragment (into machine instruction).  

** Syntax ** 
```  
disassemble
disassemble [Function]
disassemble [Address]
disassemble [Start],[End]
disassemble [Function],+[Length]
disassemble [Address],+[Length]
disassemble /m [...]
disassemble /r [...]  
```  

** Parameters **  
  
`Function`  
    Specifies the function to disassemble. If specified, the disassemble command will produce the disassembly output of the entire function.  
`Address`  
    Specifies the address inside a function to disassemble. Note that when only one address is specified, this command will disassemble the entire function that includes the given address, including the instructions above it.  
`Start/End`  
    Specifies starting and ending addresses to disassemble. If this form is used, the command won't disassemble the entire function, but only the instructions between the starting and ending addresses.  
`Length`  
    Specifies the amount of bytes to disassemble starting from the given address or function.   
`/m`  
    When this option is specified, the disassemble command will show the source lines that correspond to the disassembled instructions.  
`/r`  
    When this option is specified, the disassemble command will show the raw byte values of all disassembled instructions.   


e.g. 
```  
disassemble main  
```  

### `i` , `i[nfo]`

`i[nfo] b` -	[List breakpoints](https://ccrma.stanford.edu/~jos/stkintro/Useful_commands_gdb.html)

### [`i r`](http://visualgdb.com/gdbreference/commands/info_registers), `info registers`  

Displays the contents of general-purpose processor registers.  

#### Syntax
```  
info registers
info registers [Register name]
```  

#### Parameters

*Register name*  
    If specified, the info registers command will show the contents of a given register only. If omitted, the command will show the contents of all general-purpose CPU registers. 

e.g. `info registers eax`, `info registers cx`    


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


### `s`, `step`  
Single stepping until exit from function, which has no line number information.


### [`x`](http://visualgdb.com/gdbreference/commands/x)  

`x` displays the memory contents at a given address using the specified format.  `x` can also display memory contents of a given *register*.  

*Syntax* 
``` 
	x [Address expression]
	x/[Format] [Address expression]
	x /[Length][Format] [Address expression]  
```  

*Parameters*  

- `Address expression` - can be a specified memory address, or C/C++ expression evaluating to address, or registers (e.g. `$rbp`, and pseudoregisters (e.g. `$pc`).  
- `Length` - specifies number of elements that'll be displayed by this command `x`  
- `Format` = if specified, output format specified by  
	* o - octal
    * x - hexadecimal
    * d - decimal
    * u - unsigned decimal
    * t - binary
    * f - floating point
    * a - address
    * c - char
    * s - string
    * i - instruction  

	The following size modifiers are supported:  

    * b - byte
    * h - halfword (16-bit value)
    * w - word (32-bit value)
    * g - giant word (64-bit value)
 
 


## Possibly useful links about `gdb`  

* [disassemble command; Disassembles a specified function or a function fragment.](http://visualgdb.com/gdbreference/commands/disassemble)  
- [gdb – Assembly Language Debugging 101 from mohit.io](http://mohit.io/blog/gdb-assembly-language-debugging-101/)
- [GDB cheatsheet](http://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)

## Segmentation Fault links  

- [Buffer Overflow (vs) Buffer OverRun (vs) Stack Overflow [duplicate]](https://stackoverflow.com/questions/1144088/buffer-overflow-vs-buffer-overrun-vs-stack-overflow)
- [Segfault on stack overflow](https://stackoverflow.com/questions/81202/segfault-on-stack-overflow)
- [Segmentation Fault - C [duplicate]](https://stackoverflow.com/questions/10668504/segmentation-fault-c)
- [Memory Layout of C Programs 2.7, GeeksforGeeks](http://www.geeksforgeeks.org/memory-layout-of-c-program/)
