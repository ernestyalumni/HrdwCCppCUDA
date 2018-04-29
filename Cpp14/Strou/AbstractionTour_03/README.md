# Abstraction Mechanisms, Classes, Class Hierarchies, Templates

cf. pp. 59, Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

## Abstract Types 

*concrete types* - representation is part of their definition; they resemble built-in types.  
*abstract type* is type that completely insulates user from implementation details. 
  - to do that, decouple interface from representation, and give up genuine local variables. 
  - since we don't know anything about the representation of an abstract type (not even its size) we must allocate objects on free store (Sec. 3.2.1.2, Sec. 11.2), and access them through references or pointers (Sec. 2.2.5, 7.2, 7.7). 

1st., define interface of a class `Container` which we'll design as more abstract version of `Vector` (or `Sequence`):


```
class Container 
{
  public:
    virtual double& operator[](int) = 0;  // pure virtual function
    virtual int size() const = 0;         // const member function (Sec. 3.2.1.1)
    virtual ~Container()                  // destructor (Sec. 3.2.1.2)
      {}
};
``` 

This class is a pure interface to specific containers defined later. 
`virtual` means "maybe redefined later in a class derived from this one."  
- class derived from `Container` provides implementation for `Container` interface. 
- `=0` syntax says function is *pure virtual*; i.e. some class derived from `Container` *must* define function. 
Thus, it's not possible to define an object that's just a `Container`: `Container` can only serve as interface to a class that implements its functions (e.g. `operator[]()`, and `size()`).  
*abstract class* - class with a pure virtual function.  

`Container` can be used like this (see below for reasons why this is ok) 
``` 
void use(Container& c)
{
  const int sz {c.size()};

  for (int i {0}; i != sz; ++i)
  {
    std::cout << c[i] << '\n';
  }
}
```

*polymorphic type* - class that provides interface to a variety of other classes. cf. (Sec. 20.3.2) 

As is common for abstract classes, `Container` doesn't have a constructor. After all, it doesn't have any data to initialize. 
`Container` does have a destructor and that destructor is `virtual`. 
  - common for abstract classes to have a `virtual` destructor, because they tend to be manipulated through references or pointers, and somoen destroying a `Container` through a pointer has no idea what resources are owned by its implementation, cf. Sec. 3.2.4

A container that implements the functions required by the interface defined by the abstract class `Container` can use concrete class `Vector`: 

``` 
class Vector_container: public Container  // Vector_container implements Container
{
    Vector v;
  public:
    Vector_container(int s): v{s}
      {}
    ~Vector_container()
    {}
    
    double& operator[](int i)
    {
      return v[i];
    }

    int size() const 
    {
      return v.size();
    }
}
``` 

`:public` can be read as "is derived from" or "is a subtype of".  
Member functions `operator[]()` and `size()` *override* corresponding members in base class `Container` (Sec. 20.3.2).  Destructor (`~Vector_container()`) overrides base class destructor (`~Container()`). Note member destructor (`*Vector()`) is implicitly invoked by its class's destructor (`~Vector_container()`). 

For a function like `use(Container&)` to use `Container` in complete ignorance of implementation details, some other function will have to make an object on which it can operate. e.g. 

```
void g()
{
  Vector_container vc {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  use(vc);
}
``` 
Since `use()` doesn't know about `Vector_containers` but only knows `Container` interface, it'll work just as well for a different implementation of `Container`. 

The point is that `use(Container&)` has no idea if its argument is a `Vector_container`, `List_container`, or some other kind of container; it doesn't need to know. It can use any kind of `Container`. 
It only knows the interface defined by `Container`. Consequently, `use(Container&)` needn't be recompiled if implementation of `List_container` changes or a brand-new class derived from `Container` is used. 

The flip side of this flexibility is that objects must be manipulated through pointers or references (Sec. 3.3, Sec. 20.4). 

### Virtual Functions, `vtbl`, virtual function table

cf. pp. 67, Sec. 3.2.3 Virtual Functions. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

``` 
void use(Container& c)
{
  const int sz {c.size()};

  for (int i {0}; i != sz; ++i)
  {
    std::cout << c[i] << '\n';
  }
}
```

How is the call `c[i]` in `use()` resolved to the right `operator[]()`? 
- when `h()` calls `use()`, `List_container`'s `operator[]()` must be called. 
- when `g()` calls `use()`, `Vector_container`'s `operator[]()` must be called. 
To achieve this resolution, a `Container` object must contain information to allow it to select the right function to call at **run time**. 

The usualy implementation technique is for compiler to convert name of a virtual function into a table of pointers to functions, called *virtual function table* or `vtbl`. 
Each class with virtual functions has its own `vtbl` identifying its virtual functions. 

Functions in the `vtbl` allow object to be used correctly even when size of object and layout of its data are unknown to the caller. 
Implementation of the caller needs only to know location of pointer to `vtbl` in a `Container` and index used for each virtual function. 
Virtual call mechanism almost as time efficient as "normal function call" mechanism (within 25%). Its space overhead is 1 pointer in each object of a class with virtual functions plus 1 `vtbl` for each such class. 

### Class Hierarchies 

cf. pp. 68, Sec. 3.2.4 Class Hierarchies. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

A virtual destructor is essential for an abstract class because an object of a derived class is usually manipulated through the interface provided by its abstract base class. 
- Particularly, it (object of a derived class?) may be deleted through a pointer to a base class. Then, virtual function call mechanism ensures that proper destructor is called. That destructor then implicitly invokes destructors of its base and members. 

Class hierarchy offers 2 kinds of benefits:
* *Interface inheritance* object of derived class can be used wherever an object of a base class is required, i.e. base class acts as an interface for derived class. Example `Container` are often abstract classes. 
* *Implementation inheritance* base class provides functions or data that simplifies implementation of derived classes. Such base classes often have data members and containers. 

Concrete classes - especially classes with small representations - are much like built-in types.  
Classes in class hierarchies are different: we tend to allocate them on free store using `new`, and access them through pointers or references. 


functions returning a ptr to an object allocated on free store are dangerous. 

### Copy and Move operations 

cf. pp. 72, Sec. 3.3 Copy and Move. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

By default, objects can be copied. 

When we design a class, we must always consider if and how object might be copied. 
- for simple concrete types, memberwise copy is often exactly the right semantics for copy. 
- for some sophisticated concrete types, such as `Vector`, memberwise copy isn't the right semantics for copy, and 
- for abstract types it almost never is








