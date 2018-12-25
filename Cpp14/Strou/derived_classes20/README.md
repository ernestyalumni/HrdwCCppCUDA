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
To get runtime polymorphic behavior in C++, member functions called must be `virtual` and objects must be manipulated through ptrs or references. 
  - When manipulating an object *directly* (rather than through a ptr or reference), its exact type is known by the compiler so that run-time polymorphism isn't needed.

`std::is_polymorphic` means non-union class that declares or inherits at least 1 virtual function. 

Indeed, by default, a function that overrides a virtual funciton itself becomes `virtual`. We can, but don't have to repeat `virtual` in derived class; Stroustrup doesn't recommend repeating `virtual`; for explicit, use `override`.

### vtables, vptrs

Clearly, to implement polymorphism compiler must store some kind of type information in each object of class `Employee` and use it to call right version of virtual function `print()`.

In a typical implementation, space taken is just enough to hold a ptr (Sec. 3.2.3).
  - size of a ptr is dependent on platform. 64-bit = 8 bytes = size of ptr.

Usual implementation is for compiler to convert name of vritual function into index into a table of pointers to functions.  That table is called *virtual function table* or **`vtbl`**.

Each class with virtual functions has its own `vtbl` (I'll call it vtable) identifying its virtual functions.

For base class `A`, derived class `B`

```
class A
{
  public:
    virtual bool is_A() { return true; }

  private:
    double a_;
};

class B : public A
{
  public:
    bool is_A() { return false; }

  private:
    double b_;
}
```

A
######
vptr 8 bytes -----------> A vtable ------------>  A::is_A()
a_   8 bytes
######

B
######
vptr 8 bytes -----------> B vtable ------------> B::is_B()
b_   8 bytes
######

virtual call mechanism can be made almost as efficient as "normal function call" mechanism (within 25%), so efficiency concerns shouldn't deter anyone from using a virtual function.

Space overhead is 1 ptr in each object of a class with virtual functions, plus 1 vtable for each such class; you pay this overhead only for objects of a class with a virtual function.

Virtual function invoked from ctor or dtor reflects that object is partially constructed or partially destroyed (Sec. 22.4). It's therefore *typically bad idea to call virtual function from ctor or dtor.*

## Explicit Qualification, `::`, `A::`

cf. 20.3.3 Explicit Qualification, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Calling a function using scope resolution operator, `::`, as is done in `Manager::print()` ensures `virtual` mechanism is *not used*:

```
void Manager::print() const
{
  Employee::print(); // not a virtual call
  std::cout << "\t level " << level << '\n';
  //...
}
```

Otherwise, `Manager::print()` would suffer *infinite recursion*.

Use of a qualified name also has desirable effect that, if virtual function is also `inline` (not uncommon), then inline substitution can be used for calls specified using `::`.
  - provides efficient way to handle some important special cases in which 1 virtual function calls another for same object. 
  - because type of object is determined in call of `Manager::print()`, it need not be dynmaically determined again for resulting call of `Employee::print()`

## Override Control

cf. 20.3.4 Override Control, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

If you declare a function in a derived class that has *exactly same name and type as a virtual function in a base class*, then function in derived class *overrides the one in base class*.
  - simple rule

For larger class hierarchies, can be difficult to be sure you actually override.

Stroustrup don't (redundantly) use `virtual` for function that's meant to override.

For larger hierarchies, 
  * `virtual` - function may be override (Sec. 20.3.2)
  * `= 0` function must be `virtual` and must be overridden (Sec. 20.4)
  * `override` function meant to override virtual function in a base class (Sec. 20.3.4.1)
  * `final` function not meant to be overridden (Sec. 20.3.4.2)

### `override`

cf. 20.3.4.1 `override`, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  


In large or complicated class hierarchy with many virtual functions, it's best to use `virtual` only to introduce a new virtual function, and to use `override` on all functions intended as overriders.
  - Using `override` is a bit verbose, but clarifies programmer's intent.

`override` specifier can't be repeated in an out-of-class definition.

`override` isn't a keyword, it's a *contextual keyword*

### `final`; when to use `virtual`

cf. 20.3.4.2 `final`, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Use `virtual` for functions we want writers of derived classes to be able to define or redefine.
* can we imagine need for further derived classes?
* Does designer of derived class need to redefine function to achieve plausible aim?
* is overriding function error-prone (i.e. is it hard for overriding function to provide expected semantics of a virtual function?)

If answer is "no" to *all 3 questions*, leave function *non-`virtual`*.

Far more rarely, we have a class hierarchy that starts out with virtual functions, but after definition of a set of derived classes, 1 of the answers become "no".
Prevent users from overriding virtual functions. After using `final` for a member function, it can no longer be overridden and attempts to do so is error.
  - It also **prevents further derivation from a class**.
  - Don't blindly use `final` as optimization (non-`virtual` function is faster than `virtual` one); it affects class hierarchy design (often negatively), and performance improvements are rarely significant. Do some serious measurements before claiming efficiency improvements.
  - Use `final` where it clearly reflects class hierarchy design, i.e. reflect semantic need.

A `final` specifier isn't part of the type of a function and cannot be repeated in an out-of-class definition.
e.g.
```
class Derived : public Base
{
  void f() final; // OK if Base has a virtual f()
  void g() final; // OK if Base has a virtual g()
  // ...
};

void Derived::f() final // error: final out of class
{
  // ...
}

void g() final // OK
{
  // ...
}
```

### `using` Base Members, inheriting ctors

cf. 20.3.5 `using` Base Members. Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Functions don't overload across scopes (Sec. 12.3.3); e.g.

```
struct Base
{
  void f(int);
};

struct Derived : Base
{
  void f(double);
};

void use(Derived d)
{
  d.f(1); // call Derived::f(double)
  Base& br = d;
  br.f(1); // call Base::f(int)
}
```
This could surprise people. 
Sometimes we want overloading to ensure that best matching member function is used.
  - `using`- declarations can be used to add a function to a scope


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
* When `using`-declaration combined with private or protected derivation, it can be used to specify interfaces to some, but not all, of the facilities usually offered by a class; e.g.
```
class BB : private B // give access to B::b and B::c, but not B::a
{
  public:

    using B::b;
    using B::c;
};
```


## Pointers to Members; Pointers to Function Members, Pointers to Data Members

cf. pp. 607 Sec. 20.6 Pointers to Members, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*

Pointer to member is offset-like construct that allows programmer to indirectly refer to member of a class. 
Operators `->*`, `.*` are arguably most specialized and least used C++ operators.
- `->`, we can access member of class, `m` by naming it: `p->m`
- `->*`, we can access member that (conceptually) has its name stored in a pointer to mmeber, `ptom: p->*ptom`
This allows us to access members with their names passed as arguments.
In both cases, `p` must be a pointer to an object of an appropriate class.

Pointer to member can't be assigned to `void*` or any other ordinary pointer.
Null pointer (`nullptr`) can be assigned to pointer to member, and then represents "no member."

We can use a pointer to member to indirectly refer to member of a class.

```
class StdInterface 
{
  public:
    virtual void start() = 0;
    virtual void suspend() = 0;
    virtual void resume() = 0;
    virtual void quit() = 0;
    virtual void full_size() = 0;
    virtual void small() = 0;

    virtual ~StdInterface() {}
};
``` 

I need a pointer to member referring to `StdInterface::suspend()`, and 
pointer or reference to object I need to suspend.

```
using Pstd_mem s = &StdInterface::suspend; // pointer to suspend()

void f(StdInterface* p)
{
  PstdMem s = &StdInterface::suspend; // pointer to suspend()
  p->suspend(); // direct call
  p->*s(); // call through pointer to member
}
```
*Pointer to member* can be obtained by applying address-of operator `&` to fully qualified class member name, e.g. `&StdInterface::suspend`. 
A variable of type "pointer to member of class `X`" declared using declarator of form `X::*`.

Pointer to member `m` can be used in combination with object.
Operators `->*`, `.*` allow programmer to express such combinations:
e.g. `p->*m` binds `m` to object pointer to by `p`, and `obj.*m` binds `m` to object `obj`. 
Result can be used in accordance with `m`'s type. 
  - It's not possible to store result of `->*` or `.*` operation for later use.

A pointer to member isn't a pointer to piece of memory, the way pointer to variable or pointer to function is.
It's more like an offset into structure or index into an array.

Becuase pointer to virtual member is a kind of offset, it doesn't depend on object's location in memory; pointer to virtual member can therefore be passed between different address spaces as long as same object layout is used in both.
Like pointer to ordinary functions, pointers to non-virtual member functions can't be exchanged between address spaces.

Note that function invoked through pointer to function can be `virtual`. 
e.g. when we call `suspend()` through pointer to function, we get the right `suspend()` for object to which pointer to function is applied; this is an essential aspect of pointers to functions.

When writing an interpreter, we might use pointers to members to invoke functions presents as strings:
```
map<string, StdInterface*> variable;
map<string, Pstd_mem> operation;

void call_member(string var, string oper)
{
  (variable[var]->*operation[oper])(); // var.oper()
}
```

`static` member isn't associated with a particular object, so a pointer to a `static` member is simply an ordinary pointer; e.g.

```
class Task
{
  // ...
  static void schedule();
};

void (*p)() = &Task::schedule; // OK
void (Task::* pm)() = &Task::schedule; // error: ordinary pointer assigned
  // to pointer to member.
```

#### Pointers to Data Members, to member functions with arguments and return types

cf. pp. 609 Sec. 20.6.2 Pointers to Data Members, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*

e.g. 

```
struct C
{
  const char* val; // data member
  int i; // data member 

  // member functions

  void print(int x)
  {
    std::cout << val << x << '\n';
  }

  int f1(int);
  void f2();
  C(const char* v)
  {
    val = v;
  }
};

using Pmfi = void (C::*)(int); // pointer to member function of C taking an int
using Pm = const char* C::*; // pointer to char* data member of C
```

```
void f(C& z1, C& z2)
{
  C* p = &z2;
  Pmfi pf = &C::print;
  Pm pm = &C::val; // pointer to char* data member 

  z1.print(1);
  (z1.*pf)(2);

  z1.*pm = "nv1 ";
  p->*pm = "nv2 ";

  z2.print(3);
  (p->*pf)(4);

  pf = &C::f1; // error: return type mismatch
  pf = &C::f2; // error: argument type mismatch
  pm = &C::i; // error: type mismatch
  pm = pf; // error: type mismatch
};
```

#### Pointers to Base and Derived Members

cf. pp. 610 Sec. 20.6.3 Base and Derived Members, Ch. 20 *Derived Classes* by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*

A derived class has at least the members that it inherits from its base classes.
Often it has more.

This implies that we can safely assign a pointer to a member of a base class to a pointer to a member of a derived class.
But not the other way around (assign a pointer to a member of a derived class to a pointer to a member of a base class).
Called *contravariance*.

e.g. 

```
class Text : public StdInterface
{
  public:

    void start();
    void suspend();

    // ...

    virtual void print();

  private:

    vector s;
};
```

```
void (StdInterface::*pml)() = &Text::print; // error
void (Text::*pmt)() = &StdInterface::start; // OK
```

*Contravariance rule* appears opposite of rule that says, 
  - we can assign pointer to a derived class to pointer to its base class

Both rules exist to preserve fundamental guarantee that pointer may never point to an object that doesn't at least have properties that pointer promises.
  - In this case, `StdInterface::*` can be applied to any `StdInterface` and most such objects presumably aren't of type `Text` 
  - consequently, they don't have member `Text::print`, which we tried to initialize with `pml`

 


