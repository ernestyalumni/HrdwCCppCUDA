# Derived Classes

cf. pp. 577, Ch. 20 **Derived Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

*object-oriented programming* - its basis is the simple idea of hierarchical relationships, i.e. to express commonality between classes; *base* class or *superclass* is the common class, classes derived from that are *derived* classes or *subclasses*.

C++ language features support building new classes from existing ones:
- *implementation inheritance* to save implementation effort by sharing facilities provided by base class
- *interface inheritance* - allow different derived classes to be used interchangeably through interface provided by common base class

Interface inheritance often referred to as *run-time polymorphism* (or *dynamic polymorphism*).
  - In contrast, uniform use of classes not relate by inheritance provided by **templates** (Sec. 3.4, Ch. 23) is referred to as *compile-time polymorphism* (or *static polymorphism*)

Derived class is typically larger (and never smaller) than base class in sense it holds more data and provides more functions.






cf. pp. 245, Exploration 37 *Inheritance*, by Ray Lischner. **Exploring C++11 (Expert's Voice in C++)**.  *2nd. ed.* Apress (2013).    

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


