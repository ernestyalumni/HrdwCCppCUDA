# Heap and Stack, Memory Management, in C

[Stack Data Structure (Introduction and Program)](http://www.geeksforgeeks.org/stack-data-structure-introduction-program/)

- `stack_arr.c`, C array implementation of stack; uses `malloc` so ironically, it's a "stack" on heap memory.  cf. [Stack Data Structure (Introduction and Program)](http://www.geeksforgeeks.org/stack-data-structure-introduction-program/)   
	* Pros; easy to implement, memory not saved as pointers; Cons; no dynamic, doesn't grow and shrink depending on runtime needs  
	* `gdb` note: `gdb ./stack_arr`, then 
```  
		(gdb) break main
		(gdb) run
		(gdb) next
		(gdb) bt # back trace, look at stack
		(gdb) next
		(gdb) bt
		...  

```  
	
You literally see each function call go to the stack, with each line of code, until `No stack`.  

```   
	(gdb) next  
	20 pushed to stack  
	73		push(stack, 30);  
	(gdb) bt  
	#0  main () at stack_arr.c:73  
	(gdb) next  
	30 pushed to stack  
	75		printf("%d popped from stack\n", pop(stack));  
	(gdb) bt  
	#0  main () at stack_arr.c:75  
	(gdb) next
	[Inferior 1 (process 7142) exited normally]
	(gdb) bt
	No stack.

``` 

These are other neat commands in `gdb` to directly observe how `malloc` assigns a memory block ("chunk of memory") to a `struct Stack`, `stack` and `struct` has contiguous (memory) addresses for its members (objects):  

```  
(gdb) p &stack
$2 = (struct Stack **) 0x7fffffffddf8 # 0x7fffffffddf8 is for, correct me if I'm wrong, gdb's own variable
(gdb) x &stack
0x7fffffffddf8:	0x00602010 # you want the right-hand side value, the actual (memory) address assigned to the memory block ("chunk of memory")  

(gdb) p stack->top
$3 = 1
(gdb) p &stack->top
$4 = (int *) 0x602010
(gdb) p &stack->capacity
$5 = (unsigned int *) 0x602014
(gdb) p &stack->array[0]
$6 = (int *) 0x602030
(gdb) p &stack->array
$7 = (int **) 0x602018
(gdb) p &stack->array[1]
$8 = (int *) 0x602034
(gdb) p &stack->array[2]
$9 = (int *) 0x602038
(gdb) p &stack->array[3]
$10 = (int *) 0x60203c
(gdb) p &stack->array[4]
$11 = (int *) 0x602040

(gdb) p stack->array[2]
$14 = 0
(gdb) p stack->array[1]
$15 = 20
(gdb) p stack->array[0]
$16 = 10
(gdb) x/s 0x602030
0x602030:	"\n"
(gdb) x/s 0x602034
0x602034:	"\024"
(gdb) x/i 0x602034
   0x602034:	adc    $0x0,%al
(gdb) x/x 0x602034
0x602034:	0x14
(gdb) x/d 0x602034
0x602034:	20
(gdb) x/d 0x602040
0x602040:	0
``` 
`x/i`, `x/s`, `x/d` let's you print out instruction-level format, `s` string format, or `d` decimal format, respectively, with `x`.  


- `stack_stack_arr.c` - `struct Stack stack` should be on stack memory, only: this is the result of `(gdb) info frame`:  

```  
(gdb) p &stack
$2 = (struct Stack *) 0x7fffffffdde0
...  
(gdb) x &stack
0x7fffffffdde0:	0xffffffff

(gdb) p &stack.n_top
$19 = (int *) 0x7fffffffdde0

(gdb) x &stack.n_top
   0x7fffffffdde0:	add    (%rax),%eax

(gdb) info frame
Stack level 0, frame at 0x7fffffffde10:
 rip = 0x400728 in main (stack_stack_arr.c:110); saved rip = 0x7ffff7a31431
 source language c.
 Arglist at 0x7fffffffde00, args: 
 Locals at 0x7fffffffde00, Previous frame's sp is 0x7fffffffde10
 Saved registers:
  rbp at 0x7fffffffde00, rip at 0x7fffffffde08

```  

Also, observe  
```  
(gdb) x stack.n_top
   0x3:	Cannot access memory at address 0x3
(gdb) x &stack.n_top
   0x7fffffffdde0:	add    (%rax),%eax
(gdb) p stack.n_top
$20 = 3
```  
It appears that possibly the integer value 3, it's part of the program/text memory of the program and it's in the "beginning" of the memory allocated to the program, the *text segment*.  
 
I also wanted to observe how the stack memory, used to store these local variables, including the `struct Stack`, and how stack "grows downward" from "high address" (large address to lower value address).  
```  
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/stack_stack_arr 

Breakpoint 3, main () at stack_stack_arr.c:91
91		struct Stack stack = createStack(10);
(gdb) p stack
$36 = {n_top = 4196144, N = 0, arr = 0x400400 <_start>}
(gdb) p &stack
$37 = (struct Stack *) 0x7fffffffdde0
(gdb) p &stack.arr
$38 = (int **) 0x7fffffffdde8
(gdb) p &stack.arr[1]
$39 = (int *) 0x400404 <_start+4>
(gdb) p &stack.arr[0]
$40 = (int *) 0x400400 <_start>
(gdb) p &stack.n_top
$41 = (int *) 0x7fffffffdde0
(gdb) x &stack.n_top
   0x7fffffffdde0:	xor    %al,(%rdi)
(gdb) p &stack.N
$42 = (unsigned int *) 0x7fffffffdde4
(gdb) x &stack.N
   0x7fffffffdde4:	add    %al,(%rax)
(gdb) next
92		struct Stack* ptr_stack = &stack;
(gdb) next

Breakpoint 1, main () at stack_stack_arr.c:101
101		push(ptr_stack, 1);
(gdb) next
1 pushed to stack
102		push(ptr_stack, 2);
(gdb) p stack
$43 = {n_top = 0, N = 10, arr = 0x7fffffffdd70}
(gdb) p &stack.n_top
$44 = (int *) 0x7fffffffdde0
(gdb) p &stack.arr[0]
$45 = (int *) 0x7fffffffdd70
(gdb) p &stack.arr[1]
$46 = (int *) 0x7fffffffdd74
```  

Observe directly the size (pun intended) of *text*, *data* (initialized data segment), and *bss* (uninitialized data segment) with `size`:  

```  
size ./stack_stack_arr
   text	   data	    bss	    dec	    hex	filename
   1943	    540	      4	   2487	    9b7	./stack_stack_arr
```  

with `dec` being the sum of *text,data,bss*, i.e. *dec = text + data + bss*.  I'm assuming *hex* is the same, but in hexidecimal form (2487 = 9b7 = 9*16^2 + 11*16 + 7).  

cf. [Memory Layout of C Programs, GeeksforGeeks](http://www.geeksforgeeks.org/memory-layout-of-c-program/) introduced `size` GNU command for memory layout size; [text, data, and bss: Code and Data Size Explained](https://mcuoneclipse.com/2013/04/14/text-data-and-bss-code-and-data-size-explained/) for breakdown of `size` categories `text`, `data`, `bss` (`bss` is uninitialized data, abbreviation for "Block Started by Symbol".   

## Stack Overflow examples  

- `main_on_main.c` :  
``` 
int main() {
	main();
}  
```  
The result of running/execution is  
```  
./main_on_main 
Segmentation fault (core dumped)
```  
The problem is the recursion which fills up the stack with its function calls.  This can be explicitly shown with `gdb`:  
```  
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/main_on_main 

Program received signal SIGSEGV, Segmentation fault.
0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) bt
#0  0x00000000004004af in main () at main_on_main.c:12
(gdb) print &main
$1 = (int (*)()) 0x4004a6 <main>
(gdb) x &main
0x4004a6 <main>:	0xe5894855  
...
(gdb) break
Breakpoint 1 at 0x4004af: file main_on_main.c, line 12.

(gdb) next

Breakpoint 1, 0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) bt
#0  0x00000000004004af in main () at main_on_main.c:12
(gdb) next

Breakpoint 1, 0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) bt
#0  0x00000000004004af in main () at main_on_main.c:12
(gdb) next

Breakpoint 1, 0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) bt
#0  0x00000000004004af in main () at main_on_main.c:12
```  

Remember to look at the stack directly with `info frame`:  

```  
(gdb) info frame
Stack level 0, frame at 0x7fffffffdd80:
 rip = 0x4004af in main (main_on_main.c:12); saved rip = 0x4004b4
 source language c.
 Arglist at 0x7fffffffdd70, args: 
 Locals at 0x7fffffffdd70, Previous frame's sp is 0x7fffffffdd80
 Saved registers:
  rbp at 0x7fffffffdd70, rip at 0x7fffffffdd78
(gdb) next

Breakpoint 1, 0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) info frame
Stack level 0, frame at 0x7fffffffdd70:
 rip = 0x4004af in main (main_on_main.c:12); saved rip = 0x4004b4
 source language c.
 Arglist at 0x7fffffffdd60, args: 
 Locals at 0x7fffffffdd60, Previous frame's sp is 0x7fffffffdd70
 Saved registers:
  rbp at 0x7fffffffdd60, rip at 0x7fffffffdd68
(gdb) next

Breakpoint 1, 0x00000000004004af in main () at main_on_main.c:12
12		main();
(gdb) info frame
Stack level 0, frame at 0x7fffffffdd60:
 rip = 0x4004af in main (main_on_main.c:12); saved rip = 0x4004b4
 source language c.
 Arglist at 0x7fffffffdd50, args: 
 Locals at 0x7fffffffdd50, Previous frame's sp is 0x7fffffffdd60
 Saved registers:
  rbp at 0x7fffffffdd50, rip at 0x7fffffffdd58
```  
We see *directly* how the stack is "growing downward", saving to registers of decreasing addresses! (**!!!**)
  
- `infiniterecursion.c`  

Same thing with infinite recursion, either no base case, or calling itself over and over.  

The `gdb` strategy is this: 
- `(gdb) run`, see signal `SIGSEGV` Segmentation Fault (who exactly knows the stack is overflow?  It happens at *runtime.* I'm guessing compiler is done with its job, compiling the machine instructions already.  Address bus assigned the addresses to a chunk of memory (memory block).  Then OS must be responsible for throwing back that stack segment is full.),  
- seeing signal `SIGSEGV`, `(gdb) break`.   
- `(gdb) run` again.  Then step through, observing the stack *directly* with `next`, `info frame`, `next`, `info frame`, ...  

```  
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/infiniterecursion 

Program received signal SIGSEGV, Segmentation fault.
0x00000000004004ae in add (
    n=<error reading variable: Cannot access memory at address 0x7fffff7feffc>)
    at infiniterecursion.c:12
12	{
(gdb) break
Breakpoint 1 at 0x4004ae: file infiniterecursion.c, line 12.
(gdb) next

Program terminated with signal SIGSEGV, Segmentation fault.
The program no longer exists.
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/infiniterecursion 

Breakpoint 1, 0x00000000004004ae in add (n=0) at infiniterecursion.c:12
12	{
(gdb) info frame
Stack level 0, frame at 0x7fffffffddf0:
 rip = 0x4004ae in add (infiniterecursion.c:12); saved rip = 0x4004d5
 called by frame at 0x7fffffffde00
 source language c.
 Arglist at 0x7fffffffdde0, args: n=0
 Locals at 0x7fffffffdde0, Previous frame's sp is 0x7fffffffddf0
 Saved registers:
  rbp at 0x7fffffffdde0, rip at 0x7fffffffdde8
(gdb) next
13		return n + add(n+1);
(gdb) info frame
Stack level 0, frame at 0x7fffffffddf0:
 rip = 0x4004b1 in add (infiniterecursion.c:13); saved rip = 0x4004d5
 called by frame at 0x7fffffffde00
 source language c.
 Arglist at 0x7fffffffdde0, args: n=2
 Locals at 0x7fffffffdde0, Previous frame's sp is 0x7fffffffddf0
 Saved registers:
  rbp at 0x7fffffffdde0, rip at 0x7fffffffdde8
(gdb) next

Breakpoint 1, 0x00000000004004ae in add (n=0) at infiniterecursion.c:12
12	{
(gdb) info frame
Stack level 0, frame at 0x7fffffffddd0:
 rip = 0x4004ae in add (infiniterecursion.c:12); saved rip = 0x4004be
 called by frame at 0x7fffffffddf0
 source language c.
 Arglist at 0x7fffffffddc0, args: n=0
 Locals at 0x7fffffffddc0, Previous frame's sp is 0x7fffffffddd0
 Saved registers:
  rbp at 0x7fffffffddc0, rip at 0x7fffffffddc8
(gdb) next
13		return n + add(n+1);
(gdb) info frame
Stack level 0, frame at 0x7fffffffddd0:
 rip = 0x4004b1 in add (infiniterecursion.c:13); saved rip = 0x4004be
 called by frame at 0x7fffffffddf0
 source language c.
 Arglist at 0x7fffffffddc0, args: n=3
 Locals at 0x7fffffffddc0, Previous frame's sp is 0x7fffffffddd0
 Saved registers:
  rbp at 0x7fffffffddc0, rip at 0x7fffffffddc8
(gdb) next

Breakpoint 1, 0x00000000004004ae in add (n=32767) at infiniterecursion.c:12
12	{
(gdb) info frame
Stack level 0, frame at 0x7fffffffddb0:
 rip = 0x4004ae in add (infiniterecursion.c:12); saved rip = 0x4004be
 called by frame at 0x7fffffffddd0
 source language c.
 Arglist at 0x7fffffffdda0, args: n=32767
 Locals at 0x7fffffffdda0, Previous frame's sp is 0x7fffffffddb0
 Saved registers:
  rbp at 0x7fffffffdda0, rip at 0x7fffffffdda8
(gdb) next
13		return n + add(n+1);
(gdb) info frame
Stack level 0, frame at 0x7fffffffddb0:
 rip = 0x4004b1 in add (infiniterecursion.c:13); saved rip = 0x4004be
 called by frame at 0x7fffffffddd0
 source language c.
 Arglist at 0x7fffffffdda0, args: n=4
 Locals at 0x7fffffffdda0, Previous frame's sp is 0x7fffffffddb0
 Saved registers:
  rbp at 0x7fffffffdda0, rip at 0x7fffffffdda8
(gdb) next

Breakpoint 1, 0x00000000004004ae in add (n=0) at infiniterecursion.c:12
12	{
(gdb) info frame
Stack level 0, frame at 0x7fffffffdd90:
 rip = 0x4004ae in add (infiniterecursion.c:12); saved rip = 0x4004be
 called by frame at 0x7fffffffddb0
 source language c.
 Arglist at 0x7fffffffdd80, args: n=0
 Locals at 0x7fffffffdd80, Previous frame's sp is 0x7fffffffdd90
 Saved registers:
  rbp at 0x7fffffffdd80, rip at 0x7fffffffdd88
(gdb) next
13		return n + add(n+1);
(gdb) next

Breakpoint 1, 0x00000000004004ae in add (n=0) at infiniterecursion.c:12
12	{
(gdb) info frame
Stack level 0, frame at 0x7fffffffdd70:
 rip = 0x4004ae in add (infiniterecursion.c:12); saved rip = 0x4004be
 called by frame at 0x7fffffffdd90
 source language c.
 Arglist at 0x7fffffffdd60, args: n=0
 Locals at 0x7fffffffdd60, Previous frame's sp is 0x7fffffffdd70
 Saved registers:
  rbp at 0x7fffffffdd60, rip at 0x7fffffffdd68
```  




