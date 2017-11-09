# Exercise 13. For-Loops and Arrays of Strings  
## code that'll print out any command line arguments you pass it  

## `Segmentation Fault` found; `char* arr[]` (i.e. array of strings=char arrays, i.e. array of char arrays), outside bound  

e.g.  

```    
char *states[] = { 
	"California", "Oregon", "Washington", "Texas" }  
	
int num_states = 5;

for (i=0; i<num_states;i++) {
	printf("state %d: %s \n", i, states[i]);
}  
```  

Doing `gdb`, 
```  
gdb ./ex13_states
(gdb) run "This is a test"
state 0: California
state 1: Oregon
state 2: Washington
state 3: Texas

Program received signal SIGSEGV, Segmentation fault.
0x00007ffff7a9dfb6 in strlen () from /lib64/libc.so.6
(gdb) bt
#0  0x00007ffff7a9dfb6 in strlen () from /lib64/libc.so.6
#1  0x00007ffff7a60ab6 in vfprintf () from /lib64/libc.so.6
#2  0x00007ffff7a67709 in printf () from /lib64/libc.so.6
#3  0x00000000004006cd in main (argc=2, argv=0x7fffffffde98) at ex13_states.c:75

```  
