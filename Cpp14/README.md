# `./Cpp14/`  

## C++11 vs. C++14  

[The C++14 Standard: What You Need to Know, Dr. Dobb's](http://www.drdobbs.com/cpp/the-c14-standard-what-you-need-to-know/240169034)

compiler deduces what type.   

Reasons for *return type deduction*  
1. Use `auto` function return type to return complex type, such as iterator, 
2. refactor code   


#### `std::clog`  

cf. [`std::clog` cplusplus](http://www.cplusplus.com/reference/iostream/clog/)

`<iostream>` 

`extern ostream clog;`
standard output stream for logging, 

In terms of *static initialization order*, `clog` guaranteed to properly constructed and initialized no later than 1st time object of type `ios_bas::Init` is constructed, with inclusion of `<iostream>` counting as at least 1 initialization of such objects with `static duration`  


