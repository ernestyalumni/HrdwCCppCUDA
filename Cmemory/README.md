# `CMemory`; C Programming Language and Memory
## Where is the Memory physically located  


[`learn-c-the-hard-way-lectures`](https://github.com/zedshaw/learn-c-the-hard-way-lectures) `github` repository directly from the author, Zed Shaw.  

# Segmentation Faults  

## 

`chararr_outbnds.c`  

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
