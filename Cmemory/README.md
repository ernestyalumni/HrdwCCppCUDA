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
