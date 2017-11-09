
[`learn-c-the-hard-way-lectures`](https://github.com/zedshaw/learn-c-the-hard-way-lectures) `github` repository directly from the author, Zed Shaw.  

# `gdb`  

`print/format <what>`  
Print content of variable/memory location/register.  
  
`display/format <what>`  
Like "print", but print the information after each stepping instruction  


cf. [Learn c the hard way lecture 4](https://youtu.be/heEaKf2b1uA)  


```  
gdb  ./ex3 # get to the (gdb) prompt
(gdb) ls  # switch to your desired directory  
(gdb) run
(gdb) bt # stack trace
(gdb) break main
(gdb) run
(gdb) print age
...
```

`-g` compiler flag says to debug; provides code to `gdb` when doing, in `(gdb)` (`gdb` prompt), `list`, or `bt` (backtrace), or `step`

[Learn c the hard way lecture 4](https://youtu.be/heEaKf2b1uA?t=5m54s)

```
gdb --batch --ex run --ex bt --ex q --args ./ex3
```  
This command tells it to run all the options.  

[Shaw starts talking about Valgrand for Learn c the hard way lecture 4](https://youtu.be/heEaKf2b1uA)  

Valgrand - Track all of your memory.  Valgrand gives better error, better stack traces, heap traces, but `gdb --batch --ex run --ex bt --ex q --args ./ex3` also does it.  

[Using Lint; Learn c the hard way lecture 4](https://youtu.be/heEaKf2b1uA)  

- Looks at C code in a "static" way

If you're on Mac OS X, use AddressSanitizer; on Linux, use Valgrand.  
