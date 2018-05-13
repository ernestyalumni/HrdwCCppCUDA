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

e.g. `./Strou/except_13/StackUnwinding_eg2.cpp` 

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
 

#### `noexcept` specifier

e.g. `./Strou/except_13/noexcept_eg.cpp` 

 `noexcept` specifier - specifies whether a function could throw exceptions.

##### Syntax for `noexcept`

`noexcept` - Same as `noexcept(true)` 

Every function in C++ is either *non-throwing* or *potentially throwing*: 
* *potentially-throwing* functions are 
  * functions declared with noexcept specifier whose `expression` evaluates to `false` 
  * functions declared without noexcept specifier except for 
    * destructors, unless destructor of any potentially-constructed base or member is *potentially-throwing* 
    * default constructors, copy constructors, move constructors, defaulted on 1st declaration unless 
      * 
    * copy-assignment operators, move-assignment operators 

##### Notes for `noexcept` 

Note that **noexcept** specification on function isn't a compile-time check; it's merely a method for programmer to inform compiler whether or not function should throw exceptions. 

