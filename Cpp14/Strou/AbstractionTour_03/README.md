# Abstraction Mechanisms, Classes, Class hierarchy Tour

cf. pp. 59, Ch. 3  **A Tour of C++:Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

## Concrete Types, Concrete classes

cf. pp. 59, Sec. 3.2.1 Concrete Types Ch. 3  **A Tour of C++:Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

*concrete types* behave "just like built-in types". 

Defining characteristic of concrete type is that its representation is part of its definition. Allows implementations to be optimally efficient in time and space. Particularly,  
* places objects of concrete types on stack, in statically allocated memory, and in other objects (Sec. 6.4.2) 
* refer to objects directly (not just through pointers and references) 
* initialize objects immediately and completely (e.g. using constructors)
* copy objects (Sec. 3.3)

Representation can be private, and accessible only through member functions, but present. 
* Price to pay for having concrete types behave exactly like built-in types is if representation changes in any significant way, user must recompile. 

### Arithmetic Type; Complex number example 

Class definition itself contains only operations requiring access to representation. An industrial-strength `complex` (like the standard library one) is carefully implemented to do appropriate inlining. 

`const` specifiers on functions returning real and imaginary parts (`const` *after* function declaration) indicate these functions don't modify object for which they're called. 

### Container; RAII (Resource Acquisition Is Initialization) 

*container* - object holding a collection of elements; remember to use a constructor and destructor. 

#### Initializing Containers

cf. pp. 64, Sec. 3.2.1.3 Initializing Containers Ch. 3  **A Tour of C++:Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Container exists to hold elements, so we need convenient ways of getting elements into a container. 
2 favorites:
* *Initializer-list constructor*: Initialize with a list of elements 
* `push_back()`: Add new element at end (at the back of) the sequence 

##### Dealing with Initializer list in the constructor 

If we want to use initializer list to initialize a list, we may get this error: 

``` 
*** Error in `./VectorExample_main': free(): invalid pointer: 0x0000000000400f40 ***
======= Backtrace: =========
/lib64/libc.so.6(+0x791eb)[0x7f47599f11eb]
/lib64/libc.so.6(+0x8285a)[0x7f47599fa85a]
/lib64/libc.so.6(cfree+0x4c)[0x7f47599fe28c]
./VectorExample_main[0x400dca]
./VectorExample_main[0x400cc1]
/lib64/libc.so.6(__libc_start_main+0xf1)[0x7f4759998431]
./VectorExample_main[0x40092a]
======= Memory map: ========
00400000-00402000 r-xp 00000000 fd:02 33824422                           /home/topolo/PropD/HrdwCCppCUDA/Cpp14/Strou/AbstractionTour_03/original/VectorExample_main
00601000-00602000 r--p 00001000 fd:02 33824422                           /home/topolo/PropD/HrdwCCppCUDA/Cpp14/Strou/AbstractionTour_03/original/VectorExample_main
00602000-00603000 rw-p 00002000 fd:02 33824422                           /home/topolo/PropD/HrdwCCppCUDA/Cpp14/Strou/AbstractionTour_03/original/VectorExample_main
0117b000-011ad000 rw-p 00000000 00:00 0                                  [heap]
7f4754000000-7f4754021000 rw-p 00000000 00:00 0 
7f4754021000-7f4758000000 ---p 00000000 00:00 0 
7f4759978000-7f4759b35000 r-xp 00000000 fd:00 1973797                    /usr/lib64/libc-2.24.so
7f4759b35000-7f4759d34000 ---p 001bd000 fd:00 1973797                    /usr/lib64/libc-2.24.so
7f4759d34000-7f4759d38000 r--p 001bc000 fd:00 1973797                    /usr/lib64/libc-2.24.so
7f4759d38000-7f4759d3a000 rw-p 001c0000 fd:00 1973797                    /usr/lib64/libc-2.24.so
7f4759d3a000-7f4759d3e000 rw-p 00000000 00:00 0 
7f4759d3e000-7f4759d54000 r-xp 00000000 fd:00 1973273                    /usr/lib64/libgcc_s-6.4.1-20170727.so.1
7f4759d54000-7f4759f53000 ---p 00016000 fd:00 1973273                    /usr/lib64/libgcc_s-6.4.1-20170727.so.1
7f4759f53000-7f4759f54000 r--p 00015000 fd:00 1973273                    /usr/lib64/libgcc_s-6.4.1-20170727.so.1
7f4759f54000-7f4759f55000 rw-p 00016000 fd:00 1973273                    /usr/lib64/libgcc_s-6.4.1-20170727.so.1
7f4759f55000-7f475a05d000 r-xp 00000000 fd:00 1973950                    /usr/lib64/libm-2.24.so
7f475a05d000-7f475a25c000 ---p 00108000 fd:00 1973950                    /usr/lib64/libm-2.24.so
7f475a25c000-7f475a25d000 r--p 00107000 fd:00 1973950                    /usr/lib64/libm-2.24.so
7f475a25d000-7f475a25e000 rw-p 00108000 fd:00 1973950                    /usr/lib64/libm-2.24.so
7f475a25e000-7f475a3d6000 r-xp 00000000 fd:00 1974541                    /usr/lib64/libstdc++.so.6.0.22
7f475a3d6000-7f475a5d6000 ---p 00178000 fd:00 1974541                    /usr/lib64/libstdc++.so.6.0.22
7f475a5d6000-7f475a5e0000 r--p 00178000 fd:00 1974541                    /usr/lib64/libstdc++.so.6.0.22
7f475a5e0000-7f475a5e2000 rw-p 00182000 fd:00 1974541                    /usr/lib64/libstdc++.so.6.0.22
7f475a5e2000-7f475a5e6000 rw-p 00000000 00:00 0 
7f475a5e6000-7f475a60c000 r-xp 00000000 fd:00 1976839                    /usr/lib64/ld-2.24.so
7f475a7de000-7f475a7e2000 rw-p 00000000 00:00 0 
7f475a808000-7f475a80b000 rw-p 00000000 00:00 0 
7f475a80b000-7f475a80c000 r--p 00025000 fd:00 1976839                    /usr/lib64/ld-2.24.so
7f475a80c000-7f475a80d000 rw-p 00026000 fd:00 1976839                    /usr/lib64/ld-2.24.so
7f475a80d000-7f475a80e000 rw-p 00000000 00:00 0 
7fff12890000-7fff128b1000 rw-p 00000000 00:00 0                          [stack]
7fff128f8000-7fff128fb000 r--p 00000000 00:00 0                          [vvar]
7fff128fb000-7fff128fd000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]
Aborted (core dumped)
``` 

Naively, I did, 

``` 
VectorExample::VectorExample(std::initializer_list<double> l):
	elements_{const_cast<double*>(l.begin())},
	sz_{static_cast<int>(l.size())}
{}
``` 
Doing a `const_cast` onto a `const T*` from `std::initializer_list` maybe the problem.




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

### Copy and Move operations, Copying Containers, copy constructor, copy assignment

cf. pp. 72, Sec. 3.3 Copy and Move. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

By default, objects can be copied. 

When we design a class, we must always consider if and how object might be copied. 
- for simple concrete types, memberwise copy is often exactly the right semantics for copy. 
- for some sophisticated concrete types, such as `Vector`, memberwise copy isn't the right semantics for copy, and 
- for abstract types it almost never is

**`this`** - the name `this` is predefined in member functions and points to the object for which the member function is called. 

A copy constructor and copy assignment for a class `X` are typically declared to take an argument of type `const X&`.  

#### Move

`&&` means "rvalue reference" and is a reference to which we can bind an rvalue (Sec. 6.4.1).  
- "rvalue" roughly means "something that can appear on left-hand (right-hand?) side of an assignment" 
 	- so rvalue is - to a 1st approximation - a value that you can't assign to, such as an integer returned by a function call, and a rvalue reference is a reference to something that nobody else can assign to. The `res` local variable in `operator+()` for `Vector`s is an example. 
- move constructor *doesn't* take a `const` argument: after all, a move constructor is supposed to remove the value from its argument. 
- move operation is applied when a rvalue reference is used as an initializer or as right-hand side of an assignment.

### Resource Management 

cf. pp. 76, Sec. 3.3.3 Resource Management. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

use standard-library `std::vector` to hold `thread`s 

We can acheive *strong resource safety*, i.e. eliminate resource leaks for a general notion of a resource.  
Examples are `vector`s holding memory, **`thread`s holding system threads, `fstream`s holding file handles**.

#### Suppressing Operations, delete constructors

cf. pp. 77, Sec. 3.3.4 Suppressing Operations. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Using default copy or move for a class in a hierarchy is typically a disaster: given only a pointer to a base, we simply don't know what members the derived class has (Sec. 3.2.2), so we can't know how to copy them. 
Best thing to do is usually to *delete* the default copy and move operations.  

If you need to copy an object in a class hierarchy, write some kind of clone function (Sec. 22.2.4). 

A base class in a class hierarchy is just 1 example of an object we wouldn't want to copy. A resource handle generally can't be copied just by copying its members (Sec. 5.2, Sec. 17.2.2). 

`=delete` mechanism is general, i.e. it can be used to suppress any operation (Sec. 17.6.4). 

## Templates 

cf. pp. 78, Sec. 3.4 Templates. Ch. 3 **A Tour of C++: Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  
