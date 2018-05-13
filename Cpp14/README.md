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


#### `std::initializer_list` 

(not to be confused with `member initializer list`)

Object of type `std::initializer_list<T>` is a lightweight proxy object that provides access to any array of objects of type `const T`.  

`std::initializer_list` object is automatically constructed when: 
* *braced-init-list* used to *list-initialize* an object, where corresponding constructor accepts an `std::initializer_list` parameter 
* *braced-init-list* used as right operand of *assignment* or as *function call argument*, and corresponding assignment operator/function accepts an `std::initializer_list` parameter 
* *braced-init-list* bound to `auto`, including *ranged for loop*. 

#### `std::runtime_error` 

Defined in header `<stdexcept>` 
``` 
class runtime_error; 
```   
Defines a type of object to be thrown as exception. Reports errors that are due to events beyond the scope of the program and can not be easily predicted. 


##### Member functions (of `std::runtime_error`)

* (constructor) - constructs the exception object 

`std::runtime_error::runtime_error` 

``` 
explicit runtime_error(const std::string& what_arg);
explicit runtime_error(const char* what_arg);
``` 
Constructs the exception object with `what_arg` as explanatory string that can be accessed through `what()`. 

#### Inherited from `std::exception` for `std::runtime_error` 

##### Member functions (inherited from `std::exception` for `std::runtime_error`) 

(destructor) [ virtual] - destroys the exception object (virtual public member function of `std::exception`) 

`what` [virtual] - returns an explanatory string (virtual public member function of `std::exception`)

cf. [`std::runtime_error`](http://en.cppreference.com/w/cpp/error/runtime_error) 
 
