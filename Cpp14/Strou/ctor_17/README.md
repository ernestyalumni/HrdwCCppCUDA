# Construction, Copy and Move; Rule of 5  

cf. pp. 481, Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Difference between *move* and *copy* is that after a copy, 2 objects must have same value, whereas after a move, source of the move not required to have its original value.  

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

cf. [`ctor.cpp`]()

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

cf. pp. 484 17.2.1 Constructors and Invariants Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

*constructor* member with same name as its class; 
constructor declaration specifies an argument list (exactly as for a function), but has no return type.  
constructor's job is to initialize an object of its class

*class invariant* - something that must hold whenever a member function is called (from outside the class), and often constructor's initialization must establish this.  

class invariant stated as comments (often the case); constructor must make that true

