20180124 update  

### [`explicit` specifier](http://en.cppreference.com/w/cpp/language/explicit)  

`explicit` specfies that constructor (or conversion function) doesn't allow `implicit` conversion or `copy-initialization`; may only appear 

e.g. `explicit` used so one doesn't implicit convert from array of chars to string.  

### ['std::forward'](http://en.cppreference.com/w/cpp/utility/forward)  

```  
template< class T >
T&& forward( typename std::remove_reference<T>::type& t ) noexcept ; 

template< class T >
constexpr T&& forward( typename std::remove_reference<T>::type& t ) noexcept;

```  
1) forwards lvalues as either l values or as rvalues, depending on `T`




```  
template< class T >
T&& forward( typename std::remove_reference<T>::type&& t ) noexcept;  

template< class T >
constexpr T&& forward( typename std::remove_reference<T>::type&& t ) noexcept; 
```  

2) Forwards rvalues as rvalues and prohibits forwarding of rvalues as lvalues 

#### Perfect forwarding 

cf. [Perfect forwarding and universal references in C++, Eli Bendersky](https://eli.thegreenplace.net/2014/perfect-forwarding-and-universal-references-in-c/)





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

### rvalue references  

cf. [C++ Source, A Brief Introduction to Rvalue References, Howard E. Hinnant, Bjarne Stroustrup, and Bronek Kozicki](http://www.artima.com/cppsource/rvalue.html)  

An *lvalue reference* is formed by place an `&` after some type. 
```  
A a;
A& a_ref1 = a; 	// an lvalue reference  
```  

*rvalue reference* formed by placing an `&&` after some type.  
```  
A a;
A&& a_ref2 = a; 	// an rvalue reference 
```  

An rvalue reference behaves just like an lvalue reference except that it can bind to a temporary (an rvalue), 
whereas you can't bind a (non const) lvalue reference to an rvalue.  

- cannot bind (non const) lvalue to rvalue

```  
A& a_ref3 = A(); 	// Error! 
A&& a_ref4 = A(); 	// Ok
```  

#### `move`, `std::move`  

`move` accepts either an lvalue or rvalue argument, and return an rvalue, *without* triggering copy construction  

`move` : { lvalue, rvalue } -> { rvalue }

It's now up to client code to overload key functions on whether their argument is an lvalue or rvalue (e.g. copy constructor and assignment operator)  

When argument is lvalue, argument must be copied from.  When is an rvalue, it can safely be moved from.  

#### Perfect Forwarding (from rvalue referenceperspective, from Hinnant, Stroustrup, Kozicki)   

Imagine writing generic factory function that  
returns a `std::shared_ptr` for a newly constructed generic type.  
Obviously, factory function must accept exactly same sets of arguments as constructors of the type of objects constructed.  




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

