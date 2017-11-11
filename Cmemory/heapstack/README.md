# Heap and Stack, Memory Management, in C

[Stack Data Structure (Introduction and Program)](http://www.geeksforgeeks.org/stack-data-structure-introduction-program/)

- `stack_arr.c`, C array implementation of stack; uses `malloc` so ironically, it's a "stack" on heap memory.  cf. [Stack Data Structure (Introduction and Program)](http://www.geeksforgeeks.org/stack-data-structure-introduction-program/)   
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

``` 

