cf. [`std::array`](http://en.cppreference.com/w/cpp/container/array)

# `std::array`  

Defined in header `<array>`  

```  
template<
	class T,
	std::size_t N
> struct array;  
```  

"This container is an aggregate type with the same semantics as a struct holding a [C-style array](http://en.cppreference.com/w/cpp/language/array) `T[N]` as its only non-static data member.  

From pp. 208, Sec. 8.2.4 "Structures and Arrays" of Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup introduces the "API" or reference for standard library's `std::array` and gave this *partial*, simplified code to explain its structure/implementation by the actual standard library:  

```  
template<typename T, size_t N> 
struct array {	// simplified (see Sec. 34.2.1)  
	T elem[N]; 
	
	T* begin() noexcept { return elem; }
	const T* begin() const noexcept { return elem; }
	T* end() noexcept { return elem+N; }
	const T* end() const noexcept { return elem+N; }
	
	constexpr size_t size() noexcept; 
	
	T& operator[](size_t n) { return elem[n]; }
	const T& operator[](size_t n) const { return elem[n]; }
	
	T* data() noexcept { return elem; }
	const T* data() const noexcept { return elem; }
	
	// ... 
};  
```  
  
# `std::ostream`  

cf. [`std::ostream` in `cplusplus.com`](http://www.cplusplus.com/reference/ostream/ostream/)

`<ostream> <iostream>`  

Output stream objects can write sequences of characters and represent other kinds of data.  


# Construction, Copy and Move; Rule of 5  

cf. pp. 481, Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Difference between *move* and *copy* is that after a copy, 2 objects must have same value, whereas after a move, source of the move not required to have its original value.  


* a constructor initializing a string with string literal, e.g. 
```  
std::string s1 {"Adams"}; 
...  
std::string s2 {"Pratchett"}; 
```  

- copy constructor copying a `string` (into function argument `arg`)
- move constructor moving value of a `string` (from `arg` out of `identy()` into a temporary variable holding result of `ident(s1)`)
```  
std::string ident(std::string arg)
{
	return arg; // return string (move the value of arg out of ident() to a caller)  
}
```  
- move assignment moving value of `string` (from temporary variable holding result of `ident(s1)` into `s1`)  
```  
s1 = ident(s1);  
``` 
- copy assignment copying a `string` (from `s2` into `s1`) (l-value to l-value)
```  
s1 = s2; 
```  
- destructor releasing resources owned by `s1`, `s2`, and temporary variable holding result of `ident(s1)`  

cf. [`ctor_17/ctor.cpp`](https://github.com/ernestyalumni/HrdwCCppCUDA/blob/master/Cpp14/Strou/ctor_17/ctor.cpp)

Constructors, destructors, and copy and move operations for type are not logically separate.  We must define them as matched set, or suffer logical or performance problems.  

There are 5 situations in which an object is copied or moved:
* as source of an assignment
* as object initializer
* as function argument
* as function return value
* as exception  

```
class X {
	X(Sometype); 				// "ordinary constructor": create an object
	X();						// default constructor
	X(const X&);				// copy constructor
	X(X&&);						// move constructor
	X& operator=(const X&);		// copy assignment: clean up target and copy
	X& operator=(const X&&);	// move assignment: clean up target and move
	~X();						// destructor: clean up
	// ...

```

## `unordered_map`, `unordered_set`  




#### `std::clog`  

cf. [`std::clog` cplusplus](http://www.cplusplus.com/reference/iostream/clog/)

`<iostream>` 

`extern ostream clog;`
standard output stream for logging, 

In terms of *static initialization order*, `clog` guaranteed to properly constructed and initialized no later than 1st time object of type `ios_bas::Init` is constructed, with inclusion of `<iostream>` counting as at least 1 initialization of such objects with `static duration`  





