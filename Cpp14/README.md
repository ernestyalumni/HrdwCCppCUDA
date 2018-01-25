20180124 update  

### [`explicit` specifier](http://en.cppreference.com/w/cpp/language/explicit)  

`explicit` specfies that constructor (or conversion function) doesn't allow `implicit` conversion or `copy-initialization`; may only appear 

e.g. `explicit` used so one doesn't implicit convert from array of chars to string.  


### [Parameter pack](http://en.cppreference.com/w/cpp/language/parameter_pack), e.g. `Args&&...`  

Template parameter pack is template parameter that accepts 0 or more template arguments (non-types, types, or templates).  
Function parameter pack is a function parameter that accepts 0 or more function arguments.  

*variadic template* - template with at least 1 parameter pack.  

#### Syntax of Parameter pack   

Template parameter pack:  
* appears in *class template* and   
* *function template parameter* list  

``` 
type ...Args (optional)  
typename|class ... Args(optional), typename ...  , class ...
template < parameter-list > typename (C++17) | class ... Args (optional) 

```  
1) non-type template parameter pack with an optional name 
2) a type template parameter pack with an optional name 
3) template template parameter pack with an optional name 
4) function parameter pack with an optional name 
5) parameter pack expansion: expands to comma-separated list of 0 or more patterns.  
Pattern must include at least 1 parameter pack.  

Function parameter pack (a form of declarator, appears in function parameter list of a variadic function template)  
```  
Args ... args (optional) (4)  
```  

Parameter pack expansion (appears in body of a variadic template)  
```  
pattern ...  
```  



### ['std::thread`](http://en.cppreference.com/w/cpp/thread/thread)  

`std::thread`, defined in header `<thread>`  

#### Member functions of `std::thread`  

`operator=` - moves thread object (public member function)

##### Observers of `std::thread`  

`joinable` - checks whether thread is joinable, i.e. potentially running in parallel context (public member function)  
`get_id`   - returns the *id* of the thread (public member function)  
`native_handle` - returns underlying implementation-defined thread handle (public member function) 
`hardware_concurrency[static]` - returns number of concurrent threads supported by implementation (public static member function)  

##### Operations of `std::thread`  

`join` - waits for a thread to finish its execution (public member function)  
`detach` - permits thread to execute independently from the thread handle (public member function)  
`swap` - swaps 2 thread objects (public member function)  

