# Derived Classes

cf. pp. 577, Ch. 20 **Derived Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

*object-oriented programming* - its basis is the simple idea of hierarchical relationships, i.e. to express commonality between classes; *base* class or *superclass* is the common class, classes derived from that are *derived* classes or *subclasses*.

C++ language features support building new classes from existing ones:
- *implementation inheritance* to save implementation effort by sharing facilities provided by base class
- *interface inheritance* - allow different derived classes to be used interchangeably through interface provided by common base class

Interface inheritance often referred to as *run-time polymorphism* (or *dynamic polymorphism*).
  - In contrast, uniform use of classes not relate by inheritance provided by **templates** (Sec. 3.4, Ch. 23) is referred to as *compile-time polymorphism* (or *static polymorphism*)

Derived class is typically larger (and never smaller) than base class in sense it holds more data and provides more functions.

A class must be **defined** in order to be used as a base.

Typically, cleanest solution is for derived class to use only the public members of its base class. 

## Constructors and Destructors

cf. 20.2.2 Constructors and Destructors, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  


* Objects are constructed from bottom up (base before member, and member before derived), and destroyed top-down (derived before member, and member before base); Sec. 17.2.3
* Each class can initialize its members and bases (but not directly members or bases of its bases); Sec. 17.4.1
* Typically, destructors in a hierarchy need to be `virtual` Sec. 17.2.5
* Copy ctors of classes in a hierarchy should be used with care (if at all) to avoid slicing (Sec. 17.5.1.4)
* resolution of a virtual function call, a `dynamic_cast` or `typeid()` in a ctor or dtor reflects stage of construction and destruction (rather than the type of the yet-to-be-completed object), Sec. 22.4

## Type Fields

cf. 20.3.1 Type Fields, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  


To use derived classes as more than a convenient shorthand in declarations, we must solve the following problem: 
Given a pointer of type `Base*`, to which derived type does the object pointed to really belong?
There are 4 fundamental solutions:
1. Ensure only objects of a single type are pointed to (Sec. 3.4, Ch. 23)
2. Place type field in base class for functions to inspect
3. Use `dynamic_cast` (Sec. 22.2, 22.6)
4. Use virtual functions (Sec. 3.2.3, 20.3.2)

cf. pp. 245, Exploration 37 *Inheritance*, by Ray Lischner. **Exploring C++11 (Expert's Voice in C++)**.  *2nd. ed.* Apress (2013).    

Unless you've used `final` (Sec. 20.3.4.2), solution 1 relies on more knowledge about types involved than available to the compiler.
  - In general, it's not a good idea to try to be smarter than the type system, but (especially in combination with the use of templates), it can be used to implement homogeneous containers (e.g. standard-library `vector` and `map`) with unsurpassed performance.

Solutions 2, 3, 4 can be used to build heterogenous lists, i.e. lists of (**pointers to**) objects of several different types.
Solution 3 is a language-supported variant of Solution 2.
Solution 4 is a special type-safe variable of solution 2.

Combinations of solutions 1, 4 are particularly interesting and powerful; in almost all situations, they yield cleaner code than do solutions 2, 3.

Simple type-field solution best avoided because
 - programmer must consider every function that could conceivably need a test on the type field after a change.
 - each function using a type field, must know about representation and other details of implementation of every class derived from 1 containing type field.

## Virtual Functions

cf. 20.3.2 Virtual Functions, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Virtual functions allows for declaration of functions in a base class that can be redefined in each derived class. Compiler and linker will guarantee correct correspondence between objects and functions applied to them.

A virtual function *must* be defined for class in which it's first declared (unless it's declared to be a **pure** virtual function, Sec. 20.4)

A virtual function **can be used even if no class is derived from its class**.
Derived class that doesn't need its own version of a virtual function need not provide one.
  - When deriving a class, simply provide an appropriate function if it's needed.

Function from derived class with same name and same set of argument types as a virtual function in a base is said to *override* base class version of virtual function.
  - Furthermore, it's possible to override virtual function from a base with a more derived return type (Sec. 20.3.6)

Except where we explicitly say which version of a virtual function is called, overriding function is chosen as most appropriate for the object for which it's called.

Independently of which base class (interface) is used to access an object, we always get the same function when we use the virtual function call mechanism.

**polymorphism** - getting "the right" behavior from `Employee`'s functions independently of exactly what kind of `Employee` is actually used is called *polymorphism*.
- A type with virtual functions is called a *polymorphic type* or (more precisely) a *run-time polymorphic type*.




### Inheriting Constructors 

cf. 20.3.5.1 Inheriting Constructors, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Solve the problem by simply saying constructors should be inherited: 
```  
template<class T>
struct Vector : std::vector<T> {
  using vector<T>::vector;		// inherit constructors

  T& operator=[](size_type i) {check(i); return this->elem(i);}
  const T& operator=(size_type i) const {check(i); return this->elem(i);}

  void check(size_type i) {if (this->size()<i) throw Bad_index(i); }
};

Vector<int> v {1, 2, 3, 5, 8}; // OK: use initializer-list constructor from std::vector
```

### Use of `protected` Members

cf. 20.5.1.1 Use of `protected` Members, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Members declared `protected` are far more open to abuse than members declared `private`.  In particular, declaring data members `protected` is usually a design error.   
- placing significant amounts of data in a common class for all derived classes to use leaves that data open to corruption.  
- Fortunately, you don't have to use protected data; `private` is default in classes and is usually better choice. 

However,
`protected` is a fine way of specifying operations, protected member *functions*, for use in derived classes. 
- had implementation class been `private` in this example, further derivation would've been infeasible. 
- on the other hand, making bases providing implementation details `public` invites mistakes and misuse.  

## Access to Base Classes, `public`, `protected`, `private` inheritance, `using`-Declarations and Access control 

cf. pp. 605 Sec. 20.5.2 Use of `protected` Members, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

base class can be declared `private`, `protected`, or `public`.  For example,  
```
class X : public B { /* ... */ };
class Y : protected B { /* ... */ };
class Z : private B { /* ... */ };
```
* `public` derivation makes derived class a subtype of its base. For example, `X` is a kind of `B`.  This is the most common form of derivation.  
* `private` *inheritance*, `private` bases are most useful when defining a class by restricting the interfaces to a base, so that stronger guarantees can be provided.  
  - e.g., `B` is an implementation detail of `Z`.  
  - e.g. `Vector` of pointers template that adds type checking to its `Vector<void*>` base from Sec. 25.3 is a good example.  

Access specifier `public`, `protected`, `private` for a base class controls access to members of the base class and conversion of pointers and references from the derived class type to the base class type.  
Consider class `D` derived from base class `B`:  
* if `B` is `private` base, its public and protected members can be used only by member functions and friends of `D`.  Only friends and members of `D` can convert a `D*` to a `B*`.  

cf. pp. 606 Sec. 20.5.3 `using`-Declarations and Access Control, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

`using`-declaration can't be used to gain access to additional information; it's simply a mechanism for making accessible information more convenient to use.  On the other hand, once access is available, it can be granted to other users.  


