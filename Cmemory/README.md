# `CMemory`; C Programming Language and Memory
## Where is the Memory physically located  


[`learn-c-the-hard-way-lectures`](https://github.com/zedshaw/learn-c-the-hard-way-lectures) `github` repository directly from the author, Zed Shaw.  

## Pointers vs. Arrays  

`ptr_vs_arr.c`  

Consider  
```  
char arr[] = "don't panic\n";  
char* ptr = arr; 
```  
Notice that while `ptr` points to the first element of `arr`, the address of ptr itself is not the address of `arr[0]`.  `ptr` is a variable.  

```  
(gdb) p &arr
$22 = (char (*)[13]) 0x7fffffffde01
(gdb) p &ptr
$23 = (char **) 0x7fffffffde18
(gdb) x arr
0x7fffffffde01:	0x276e6f64
(gdb) x ptr
0x7fffffffde01:	0x276e6f64
(gdb) print arr
$24 = "don't panic\n"
(gdb) print ptr
$25 = 0x7fffffffde01 "don't panic\n"
(gdb) p &arr[0]
$26 = 0x7fffffffde01 "don't panic\n"
(gdb) p &arr[1]
$27 = 0x7fffffffde02 "on't panic\n"
(gdb) p &arr[2]
$28 = 0x7fffffffde03 "n't panic\n"
(gdb) p &ptr[0]
$29 = 0x7fffffffde01 "don't panic\n"
(gdb) p &ptr[1]
$30 = 0x7fffffffde02 "on't panic\n"
(gdb) p &ptr[2]
$31 = 0x7fffffffde03 "n't panic\n"  
```  
Clearly, while pointer arithmetic and indexing is equivalent for arrays and pointers (both are equipped with these operations, operators), the difference is how `ptr` is a variable located at `0x7fffffffde18`, not at `0x7fffffffde01`.  

### How `ptrs` and arrays are different: 1 way is assignment `=`  

#### Example of `gdb` usage to pinpoint the machine/assembly instructions for these particular lines  

This is for `ptr_vs_arr.c`; `ptr_vs_arr_onlyassign.c` has only the assignment code lines we desire (for reference).  

Strategy: 1st., look at your code you written in C.  You have the corresponding lines (line number) you can look up on your IDE or even text editor.  

You can set break points with `break` *`line_number`*.  Set break points surrounding your desired code.  e.g.  
```
(gdb) break main  
(gdb) run 
(gdb) next 
...  
(gdb) break 23  
(gdb) next
(gdb) break next
```
Then you can either do `disassemble` and do `break *main + ` *offset* to make breaks off the assembly code, where *offset* is the offset you see on the left-hand side, e.g.   
```  
   0x00000000004005bc <+198>:	movzbl -0x89(%rbp),%eax
   0x00000000004005c3 <+205>:	mov    %al,-0x11(%rbp)
   0x00000000004005c6 <+208>:	mov    -0x10(%rbp),%rax
   0x00000000004005ca <+212>:	movzbl 0x7(%rax),%eax
   0x00000000004005ce <+216>:	mov    %al,-0x12(%rbp)
=> 0x00000000004005d1 <+219>:	mov    $0x0,%eax  
```  

or do `i b` (list breakpoints) and see the break points you made from the line code, and observe their corresponding addresses.  Thus, you can do this, give `disassemble` a range of addresses, from *start* to *end*, to look up:   


```      
	// disassemble START, END
	(gdb) i b  
...
6       breakpoint     keep y   0x00000000004005d1 in main at ptr_vs_arr.c:33
7       breakpoint     keep y   0x00000000004005bc in main at ptr_vs_arr.c:30

(gdb) disassemble 0x4005bc, 0x4005d1
Dump of assembler code from 0x4005bc to 0x4005d1:
   0x00000000004005bc <main+198>:	movzbl -0x89(%rbp),%eax
   0x00000000004005c3 <main+205>:	mov    %al,-0x11(%rbp)
   0x00000000004005c6 <main+208>:	mov    -0x10(%rbp),%rax
   0x00000000004005ca <main+212>:	movzbl 0x7(%rax),%eax
   0x00000000004005ce <main+216>:	mov    %al,-0x12(%rbp)
End of assembler dump.  
```

and so 
`char a = array_place[7];`, assignment `=` is done, here, instruction-wise, as follows:  

* `movzbl -0x89(%rbp),%eax` - take 8th character, offset 8, to get value, with %eax register, there
* `mov    %al,-0x11(%rbp)` - `mov` contents `-0x11(%rbp)` into register `%al` 

`char b = ptr_place[7];` assignment `=` is done, here, instruction-wise, as follows:  
* `mov    -0x10(%rbp),%rax` - 1st., copy value of the pointer (which holds an address), into `%rax` register  
* `movzbl 0x7(%rax),%eax`  -  off that address in register `%rax`, offset by 7 and get value there with `%eax` register/instruction  
* `mov    %al,-0x12(%rbp)` - `mov` contents `-0x12(%rbp)` into register `%al`.  

As a sanity check, do the same with `ptr_vs_arr_onlyassign.c`.  


`ptr_vs_arr_more.c` explicitly shows the "A graphical explanation" in [Eli Bendersky's webpage](https://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c).  



# Segmentation Faults  

cf. [Debugging Segmentation Faults and Pointer Problems, CProgramming](https://www.cprogramming.com/debugging/segfaults.html)

"When your program runs, it has access to certain portions of memory. First, you have local variables in each of your functions; these are stored in the stack. Second, you may have some memory, allocated during runtime (using either malloc, in C, or new, in C++), stored on the heap (you may also hear it called the "free store"). Your program is only allowed to touch memory that belongs to it -- the memory previously mentioned. Any access outside that area will cause a segmentation fault."  

"There are 4 common mistakes that lead to segmentation faults:"  
* dereferencing NULL, 
* dereferencing an uninitialized pointer, 
* dereferencing a pointer that has been freed (or deleted, in C++) or that has gone out of scope (in the case of arrays declared in functions), and 
* writing off the end of an array.

"A 5th way of causing a segfault is a recursive function that uses all of the stack space. On some systems, this will cause a "stack overflow" report, and on others, it will merely appear as another type of segmentation fault."

"The strategy for debugging all of these problems is the same:  
- load the core file into GDB, 
- do a backtrace, 
- move into the scope of your code, and 
- list the lines of code that caused the segmentation fault.   
 


## on stack memory, no `malloc`, uninitialized pointers/arrays  


In this case, uninitialized pointer to a `struct Stack`  

```  
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/stack_stack_arr 

Program received signal SIGSEGV, Segmentation fault.
0x000000000040050e in createStack (N=10) at stack_stack_arr.c:32
32		stack->N = N;
(gdb) print stack
$1 = (struct Stack *) 0x0

```  

Printing out `stack` reveals that it points to memory address `0x0` (the `0x` indicates that the value following it is in hexadecimal, traditional for printing memory addresses).  
The address `0x0` is invalid -- in fact, it's `NULL`. If you dereference a pointer that stores the location `0x0` then you'll definitely get a segmentation fault, just as we did.

cf. `CMemory/heapstack/stack_stack_arr_fail.c`  

## Buffer Overflows (stack overflows)

### blow stack by using more memory than is available to this thread 

- e.g. `Cmemory/heapstack/main_on_main.c`  

```  
int main() 
{
	main();
}  
```  

Also, similarly (fill up stack), *infinite recursion* (no base case, or just no stopping of computation)  

- e.g. `Cmemory/heapstack/infiniterecursion.c`   

```  
int add(int n)
{
	return n + add(n+1);
}  

int main(){
	add(2);
}
```  

### `ex15.c`  



## Given a C "string" or `char[]`, `printf` an element that is outside its "length", found at run-time (because it compiles) when OS uses `libstr count`  

### `chararr_outbnds.c`  

```  
(gdb) print arr_states[0]
$9 = 0x400590 "123"
(gdb) print arr_states[1]
$10 = 0x400594 "124"
(gdb) print arr_states[2]
$11 = 0x400598 "125"
(gdb) print arr_states[3]
$12 = 0x40059c "126"
(gdb) print arr_states[4]
$13 = 0x7fffffffdef0 "\001"
(gdb) print arr_states[5]
$14 = 0x5 <error: Cannot access memory at address 0x5>  
```  
But otherwise it exits normally.  

When adding this `for`-loop, using **`printf`**, 

```  
	char* arr_states[] = { "123", "124", "125", "126" };
	int num_of_states = 6; // 6 will guarantee Segmentation Fault, outside of the End of Strings \00


	for (i=0; i<num_of_states;i++){
		printf("state %d %s \n", 
			i, 
			arr_states[i]);  // it'll NOT exit normally for num_of_states = 5,6; segmentation fault
	}	 
```  
Segmentation Fault.  Going to `gdb`, doing `gdb ./chararr_outbnds`, `(gdb) break main`, `(gdb) run`, `(gdb) next` ... 

```  
(gdb) print arr_states[4]
$2 = 0x7fffffffdef0 "\001"
```  
That's the end of the `char * arr[]`, `string=char arr` of `"\001"`, itself a `string=char arr`.   
```  
(gdb) print arr_states[5]
$3 = 0x400000006 <error: Cannot access memory at address 0x400000006>
```  

Nevertheless, 
```  
gdb) next
state 3 126 
27		for (i=0; i<num_of_states;i++){
(gdb) next
28			printf("state %d %s \n", 
(gdb) next
state 4  
27		for (i=0; i<num_of_states;i++){
(gdb) next

Program received signal SIGSEGV, Segmentation fault.
strlen () at ../sysdeps/x86_64/strlen.S:106
106		movdqu	(%rax), %xmm4

```  

## [more Segmentation Faults](https://en.wikipedia.org/wiki/Segmentation_fault)  

The operating system (OS) is running the program (its instructions).  Only from the hardware, with [memory protection](https://en.wikipedia.org/wiki/Memory_protection), with the OS be signaled to a memory access violation, such as writing to read-only memory or writing outside of allotted-to-the-program memory, i.e. data segments.  On x86_64 computers, this [general protection fault](https://en.wikipedia.org/wiki/General_protection_fault) is initiated by protection mechanisms from the hardware (processor).  From there, OS can signal the fault to the (running) process, and stop it (abnormal termination) and sometimes core dump.      

For [virtual memory](https://en.wikipedia.org/wiki/Virtual_memory), the memory addresses are mapped by program called *virtual addresses* into *physical addresses* and the OS manages virtual addresses space, hardcare in the CPU called memory management unit (*MMU*) translates virtual addresses to physical addresses, and kernel manages memory hierarchy (eliminating possible overlays).  In this case, it's the *hardware* that detects an attempt to refer to a non-existent segment, or location outside the bounds of a segment, or to refer to location not allowed by permissions for that segment (e.g. write on read-only memory).   

- 'derefnullptr.c' - point to `NULL`, dereference pointer -> maybe cause Segmentation Fault (not guaranteed by C/C++ standard, and dependent upon if you can write to memory at 0 (`0x00`)).   

```  
#include <stdio.h> // printf, NULL
... 
char *cptr = NULL; // no SegFault yet
char chout = *cptr; // Segmentation Fault occurs here  
```  

	* `gdb` debugging results:   
```  
(gdb) disassemble main
Dump of assembler code for function main:
   0x00000000004004a6 <+0>:	push   %rbp
   0x00000000004004a7 <+1>:	mov    %rsp,%rbp
   0x00000000004004aa <+4>:	movq   $0x0,-0x8(%rbp)
   0x00000000004004b2 <+12>:	mov    -0x8(%rbp),%rax
=> 0x00000000004004b6 <+16>:	movzbl (%rax),%eax
   0x00000000004004b9 <+19>:	mov    %al,-0x9(%rbp)
   0x00000000004004bc <+22>:	mov    $0x0,%eax
   0x00000000004004c1 <+27>:	pop    %rbp
   0x00000000004004c2 <+28>:	retq  	

(gdb) break main
Breakpoint 1 at 0x4004aa: file derefnullptr.c, line 17.
(gdb) break *main+27
Breakpoint 2 at 0x4004c1: file derefnullptr.c, line 20.
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/derefnullptr 

Breakpoint 1, main () at derefnullptr.c:17
17		char *cptr = NULL;  

(gdb) p cptr  // we're pointing to 0x0
$1 = 0x0
(gdb) p &cptr
$2 = (char **) 0x7fffffffde18

(gdb) c
Continuing.

Program received signal SIGSEGV, Segmentation fault.
0x00000000004004b6 in main () at derefnullptr.c:18
18		char chout = *cptr;  // Segmentation Fault occurs here 

(gdb) x/10x $rax
0x0:	Cannot access memory at address 0x0
(gdb) x/10x $eax
0x0:	Cannot access memory at address 0x0
(gdb) x/10x $rbp
0x7fffffffde20:	0x004004d0	0x00000000	0xf7a31431	0x00007fff
0x7fffffffde30:	0x00040000	0x00000000	0xffffdf08	0x00007fff
0x7fffffffde40:	0xf7b9a188	0x00000001
```  
Accessing memory at address `0x0` is problematic.  But I think the problem is the `movzbl (%rax),%eax`, (note `movzbl` is `mov`, but taking into account +,- signs).  

- 'derefwildptr.c' - wild ptr (uninitialized ptr) may point to memory that may or may not exist, and may or may not be readable or writable.  Reading from a wild ptr may result in random data but no segmentation fault.   

```  
int main() {
	char *cptr;  
	*cptr;
	char chout = *cptr;  // Segmentation Fault occurs here 
} 
```  

	* `gdb` debugging results; strategy I took was to set up break points directly at the line number of the code itself, and after looking at `disassemble main` and seeing where `ret` instruction occurred, and other "places of interest" (such as a function call with `call` instruction):  

```  
(gdb) b 18
Breakpoint 1 at 0x4004aa: file derefwildptr.c, line 18.
(gdb) b 19
Note: breakpoint 1 also set at pc 0x4004aa.
Breakpoint 2 at 0x4004aa: file derefwildptr.c, line 19.
(gdb) b 20
Note: breakpoints 1 and 2 also set at pc 0x4004aa.
Breakpoint 3 at 0x4004aa: file derefwildptr.c, line 20.
(gdb) b *main+19
Breakpoint 4 at 0x4004b9: file derefwildptr.c, line 22.

...

(gdb) r
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/derefwildptr 

Breakpoint 1, main () at derefwildptr.c:20
20		char chout = *cptr;  // Segmentation Fault occurs here 
(gdb) p cptr
$3 = 0x0
(gdb) p &cptr
$4 = (char **) 0x7fffffffde18
(gdb) disassemble main
Dump of assembler code for function main:
   0x00000000004004a6 <+0>:	push   %rbp
   0x00000000004004a7 <+1>:	mov    %rsp,%rbp
=> 0x00000000004004aa <+4>:	mov    -0x8(%rbp),%rax
   0x00000000004004ae <+8>:	movzbl (%rax),%eax
   0x00000000004004b1 <+11>:	mov    %al,-0x9(%rbp)
   0x00000000004004b4 <+14>:	mov    $0x0,%eax
   0x00000000004004b9 <+19>:	pop    %rbp
   0x00000000004004ba <+20>:	retq   
End of assembler dump.
(gdb) i r
rax            0x4004a6	4195494
rbx            0x0	0
rcx            0x0	0
rdx            0x7fffffffdf18	140737488346904
rsi            0x7fffffffdf08	140737488346888
rdi            0x1	1
rbp            0x7fffffffde20	0x7fffffffde20
rsp            0x7fffffffde20	0x7fffffffde20
r8             0x400530	4195632
r9             0x7ffff7de81b0	140737351942576
r10            0xc	12
r11            0x1	1
r12            0x4003b0	4195248
r13            0x7fffffffdf00	140737488346880
r14            0x0	0
r15            0x0	0
rip            0x4004aa	0x4004aa <main+4>
eflags         0x246	[ PF ZF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) s

Program received signal SIGSEGV, Segmentation fault.
0x00000000004004ae in main () at derefwildptr.c:20
20		char chout = *cptr;  // Segmentation Fault occurs here 
(gdb) p cptr
$5 = 0x0
(gdb) p &cptr
$6 = (char **) 0x7fffffffde18
(gdb) disassemble main
Dump of assembler code for function main:
   0x00000000004004a6 <+0>:	push   %rbp
   0x00000000004004a7 <+1>:	mov    %rsp,%rbp
   0x00000000004004aa <+4>:	mov    -0x8(%rbp),%rax
=> 0x00000000004004ae <+8>:	movzbl (%rax),%eax
   0x00000000004004b1 <+11>:	mov    %al,-0x9(%rbp)
   0x00000000004004b4 <+14>:	mov    $0x0,%eax
   0x00000000004004b9 <+19>:	pop    %rbp
   0x00000000004004ba <+20>:	retq   
End of assembler dump.
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0x0	0
rdx            0x7fffffffdf18	140737488346904
rsi            0x7fffffffdf08	140737488346888
rdi            0x1	1
rbp            0x7fffffffde20	0x7fffffffde20
rsp            0x7fffffffde20	0x7fffffffde20
r8             0x400530	4195632
r9             0x7ffff7de81b0	140737351942576
r10            0xc	12
r11            0x1	1
r12            0x4003b0	4195248
r13            0x7fffffffdf00	140737488346880
r14            0x0	0
r15            0x0	0
rip            0x4004ae	0x4004ae <main+8>
eflags         0x10246	[ PF ZF IF RF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
```  

The problem seemed to be pinpointed to pointer `cptr`, (see `p cptr`) and how it's pointing to invalid (memory) address `0x0`, and register RAX, having to move the contents of this invalid (memory) address with the `movzbl` instruction.   

That the problem occurs at this instruction level, `movzbl (%rax),%eax` as further evidenced as compiling and running only (i.e. commenting out the line `char chout = *cptr`):  
```  
int main() {
	char *cptr;
	*cptr;
}
``` 
does *NOT* trigger a Segmentation Fault.  

- 'derefdanglingptr.c'  

``` 
char *cptr = malloc(10*sizeof(char)); 
free(cptr);
char chout = *cptr;  
``` 


Dangling ptr (freed ptr, which points to memory that has been freed/deallocated/deleted) may point to memory that may or may not exist and may or may not be readable or writable.  Reading from a dangling ptr may result in valid data for a while, and then random data as it's overwritten.    
  
I didn't obtain a Segmentation Fault in this case.  We can take a look with `gdb` about what's going on, focusing on the RAX register.  

The strategy is to use `continue c` to go from break point to break point, set from looking at the line number of the code and looking at `disasemble main`; and at each "step through", look at changes in `i r` (`info registers`) and `p cptr` and `p &cptr`, and `bt` (`backtrace`) to check "where we are":  

```  
(gdb) r
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/derefdanglingptr 

Breakpoint 1, main () at derefdanglingptr.c:18
18		char *cptr = malloc(10*sizeof(char)); // sizeof(char) = 1 anyways 
(gdb) p cptr
$1 = 0x0
(gdb) p &cptr
$2 = (char **) 0x7fffffffde08
(gdb) i r
rax            0x400546	4195654
rbx            0x0	0
rcx            0x0	0
rdx            0x7fffffffdf08	140737488346888
rsi            0x7fffffffdef8	140737488346872
rdi            0x1	1
rbp            0x7fffffffde10	0x7fffffffde10
rsp            0x7fffffffde00	0x7fffffffde00
r8             0x4005f0	4195824
r9             0x7ffff7de81b0	140737351942576
r10            0xc	12
r11            0x1	1
r12            0x400450	4195408
r13            0x7fffffffdef0	140737488346864
r14            0x0	0
r15            0x0	0
rip            0x40054e	0x40054e <main+8>
eflags         0x206	[ PF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) bt
#0  main () at derefdanglingptr.c:18  

...  

(gdb) c
Continuing.

Breakpoint 7, main () at derefdanglingptr.c:19
19		free(cptr);
(gdb) i r
rax            0x602010	6299664
rbx            0x0	0
rcx            0x7ffff7dd1ae0	140737351850720
rdx            0x602010	6299664
rsi            0x602020	6299680
rdi            0x7ffff7dd1ae0	140737351850720
rbp            0x7fffffffde10	0x7fffffffde10
rsp            0x7fffffffde00	0x7fffffffde00
r8             0x602000	6299648
r9             0x0	0
r10            0x602010	6299664
r11            0x246	582
r12            0x400450	4195408
r13            0x7fffffffdef0	140737488346864
r14            0x0	0
r15            0x0	0
rip            0x40055c	0x40055c <main+22>
eflags         0x206	[ PF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) bt
#0  main () at derefdanglingptr.c:19
(gdb) p cptr
$5 = 0x602010 ""
(gdb) p &cptr
$6 = (char **) 0x7fffffffde08

(gdb) c
Continuing.

Breakpoint 8, main () at derefdanglingptr.c:21
21		char chout = *cptr;   
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000400546 <+0>:	push   %rbp
   0x0000000000400547 <+1>:	mov    %rsp,%rbp
   0x000000000040054a <+4>:	sub    $0x10,%rsp
   0x000000000040054e <+8>:	mov    $0xa,%edi
   0x0000000000400553 <+13>:	callq  0x400440 <malloc@plt>
   0x0000000000400558 <+18>:	mov    %rax,-0x8(%rbp)
   0x000000000040055c <+22>:	mov    -0x8(%rbp),%rax
   0x0000000000400560 <+26>:	mov    %rax,%rdi
   0x0000000000400563 <+29>:	callq  0x400430 <free@plt>
=> 0x0000000000400568 <+34>:	mov    -0x8(%rbp),%rax
   0x000000000040056c <+38>:	movzbl (%rax),%eax
   0x000000000040056f <+41>:	mov    %al,-0x9(%rbp)
   0x0000000000400572 <+44>:	mov    $0x0,%eax
   0x0000000000400577 <+49>:	leaveq 
   0x0000000000400578 <+50>:	retq   
End of assembler dump.
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0x7ffff7dd1a00	140737351850496
rdx            0x0	0
rsi            0x7ffff7dd1ae8	140737351850728
rdi            0xffffffff	4294967295
rbp            0x7fffffffde10	0x7fffffffde10
rsp            0x7fffffffde00	0x7fffffffde00
r8             0x602010	6299664
r9             0x0	0
r10            0x8bc	2236
r11            0x7ffff7a97240	140737348465216
r12            0x400450	4195408
r13            0x7fffffffdef0	140737488346864
r14            0x0	0
r15            0x0	0
rip            0x400568	0x400568 <main+34>
eflags         0x206	[ PF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) c
Continuing.

Breakpoint 4, 0x000000000040056c in main () at derefdanglingptr.c:21
21		char chout = *cptr;   
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000400546 <+0>:	push   %rbp
   0x0000000000400547 <+1>:	mov    %rsp,%rbp
   0x000000000040054a <+4>:	sub    $0x10,%rsp
   0x000000000040054e <+8>:	mov    $0xa,%edi
   0x0000000000400553 <+13>:	callq  0x400440 <malloc@plt>
   0x0000000000400558 <+18>:	mov    %rax,-0x8(%rbp)
   0x000000000040055c <+22>:	mov    -0x8(%rbp),%rax
   0x0000000000400560 <+26>:	mov    %rax,%rdi
   0x0000000000400563 <+29>:	callq  0x400430 <free@plt>
   0x0000000000400568 <+34>:	mov    -0x8(%rbp),%rax
=> 0x000000000040056c <+38>:	movzbl (%rax),%eax
   0x000000000040056f <+41>:	mov    %al,-0x9(%rbp)
   0x0000000000400572 <+44>:	mov    $0x0,%eax
   0x0000000000400577 <+49>:	leaveq 
   0x0000000000400578 <+50>:	retq   
End of assembler dump.
(gdb) i r
rax            0x602010	6299664
rbx            0x0	0
rcx            0x7ffff7dd1a00	140737351850496
rdx            0x0	0
rsi            0x7ffff7dd1ae8	140737351850728
rdi            0xffffffff	4294967295
rbp            0x7fffffffde10	0x7fffffffde10
rsp            0x7fffffffde00	0x7fffffffde00
r8             0x602010	6299664
r9             0x0	0
r10            0x8bc	2236
r11            0x7ffff7a97240	140737348465216
r12            0x400450	4195408
r13            0x7fffffffdef0	140737488346864
r14            0x0	0
r15            0x0	0
rip            0x40056c	0x40056c <main+38>
eflags         0x206	[ PF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) p &cptr
$9 = (char **) 0x7fffffffde08
(gdb) p cptr
$10 = 0x602010 ""
...  

```   

So at least we can conclude *definitively*  that in this case, for `movzbl`, RAX doesn't have to deal with invalid memory address `0x0`.  

- `derefnullfptr.c` - more on ptr, initialized or pointing to null, 0.  

`null pointer` has a value reserved to indicate that the pointer does not refer to a valid object.  This, for C, is `NULL` *and* `0`, the integer literal constant zero.     

Null ptr shouldn't be confused with uninitialized ptr: A null ptr is guaranteed to compare unequal to any ptr that points to a valid object.  However, depending on language, uninitialized ptr may not gurantee to point to a valid object.  cf. [wikipedia, "Null pointer"](https://en.wikipedia.org/wiki/Null_pointer)

```  
int main() {
	float *fptr = 0; // doesn't "initialize" *fptr to 0, instead, C reads it as it does NULL; // this line alone still compiles	
	// *fptr; // this line also DOES NOT trigger a Segmentation Fault 
	*fptr = 10.f;  // results in Segmentation Fault  
//	float fout = *fptr;  // results in Segmentation Fault 
}   
```   

Again, using `gdb`, with the strategy of taking a look at `disassemble main` (or `disassemble ` *functionofinterest*), and looking at code directly (and gathering the line numbers for the code, of interest), and setting break points (`b ` *linenumberofcode* and `b *main + ` *integeroffsetseenindisassemblefunction*), and using `continue c`, :

```  
(gdb) c
Continuing.

Breakpoint 3, main () at derefnullfptr.c:19
19		*fptr = 10.f;  // results in Segmentation Fault  
(gdb) disassemble main
Dump of assembler code for function main:
   0x00000000004004a6 <+0>:	push   %rbp
   0x00000000004004a7 <+1>:	mov    %rsp,%rbp
   0x00000000004004aa <+4>:	movq   $0x0,-0x8(%rbp)
=> 0x00000000004004b2 <+12>:	mov    -0x8(%rbp),%rax
   0x00000000004004b6 <+16>:	movss  0xa2(%rip),%xmm0        # 0x400560
   0x00000000004004be <+24>:	movss  %xmm0,(%rax)
   0x00000000004004c2 <+28>:	mov    $0x0,%eax
   0x00000000004004c7 <+33>:	pop    %rbp
   0x00000000004004c8 <+34>:	retq   
End of assembler dump.
(gdb) p fptr
$3 = (float *) 0x0
(gdb) p &fptr
$4 = (float **) 0x7fffffffde18
(gdb) i r
rax            0x4004a6	4195494
rbx            0x0	0
rcx            0x0	0
rdx            0x7fffffffdf18	140737488346904
rsi            0x7fffffffdf08	140737488346888
rdi            0x1	1
rbp            0x7fffffffde20	0x7fffffffde20
rsp            0x7fffffffde20	0x7fffffffde20
r8             0x400540	4195648
r9             0x7ffff7de81b0	140737351942576
r10            0xc	12
r11            0x1	1
r12            0x4003b0	4195248
r13            0x7fffffffdf00	140737488346880
r14            0x0	0
r15            0x0	0
rip            0x4004b2	0x4004b2 <main+12>
eflags         0x246	[ PF ZF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0

... 

(gdb) c
Continuing.

Program received signal SIGSEGV, Segmentation fault.
0x00000000004004be in main () at derefnullfptr.c:19
19		*fptr = 10.f;  // results in Segmentation Fault  
(gdb) disassemble main
Dump of assembler code for function main:
   0x00000000004004a6 <+0>:	push   %rbp
   0x00000000004004a7 <+1>:	mov    %rsp,%rbp
   0x00000000004004aa <+4>:	movq   $0x0,-0x8(%rbp)
   0x00000000004004b2 <+12>:	mov    -0x8(%rbp),%rax
   0x00000000004004b6 <+16>:	movss  0xa2(%rip),%xmm0        # 0x400560
=> 0x00000000004004be <+24>:	movss  %xmm0,(%rax)
   0x00000000004004c2 <+28>:	mov    $0x0,%eax
   0x00000000004004c7 <+33>:	pop    %rbp
   0x00000000004004c8 <+34>:	retq   
End of assembler dump.
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0x0	0
rdx            0x7fffffffdf18	140737488346904
rsi            0x7fffffffdf08	140737488346888
rdi            0x1	1
rbp            0x7fffffffde20	0x7fffffffde20
rsp            0x7fffffffde20	0x7fffffffde20
r8             0x400540	4195648
r9             0x7ffff7de81b0	140737351942576
r10            0xc	12
r11            0x1	1
r12            0x4003b0	4195248
r13            0x7fffffffdf00	140737488346880
r14            0x0	0
r15            0x0	0
rip            0x4004be	0x4004be <main+24>
eflags         0x10246	[ PF ZF IF RF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0

```  

We clearly see that the problem is at what register RAX has to deal with; trying to open up the contents at invalid (memory) address `0x0`.  

For completeness, the right way to do this is this:  

```  
#include <stdio.h>

int main() {
	float fvar = 20.f; 
	float *fptr = &fvar; 
	*fptr = 10.f; 
	printf(" fvar, fptr : %f, %f", fvar, *fptr); // 10.f, 10.f, as expected.  
```



# Registers  

cf. [X86-64 Architecture Guide](http://cons.mit.edu/sp17/x86-64-architecture-guide.html)  

| Register | Purpose | Saved across calls |   
| :------- | ------- | :----------------- |   
| `%rax`   | temp register; return value | No |  
| `%rsp`   | stack pointer | Yes | 
| `%rbp`   | callee-saved; base pointer | Yes |  
| `%r10-r11` | temporary | No | 
| `%r12-r15` | callee-saved registers | Yes |  

 

