Do `gcc -Wall -g ex12_break.c -o ex12_break`,  
Do `./ex12_break`  

```  
The size of an int: 4
The size of areas (int[]): 20
The number of ints in areas: 5
The first area is 10, the 2nd 12.

 10th area is 41661489 . 

 10th area, assigned to another int, is 41661489 . 

 Addresses of areas 0,1,10, and assigned int to 10th area, : 1932685856 1932685860 1932685896 1932685848 . 
The size of a char: 1
The size of name (char[]): 4
the number of chars: 4
The size of full_name (char[]): 11  # originally 12
The number of chars: 11 # originally 12
name="Zed" and full_name="Zed A. Shaw�@"
```  

Notice the address for `areas[10]` is where it's expected: `1932685896 - 1932685856 = 4*10`.  For end of a `char[]`, not including `\0`, then we obtain strange looking strings.  

# `gdb` on `./ex12_break`  

```  
gdb ./ex12_break
(gdb) 
```

Also,  
```
gdb --batch --ex run --ex bt --ex q --args ./ex12_break  
```
