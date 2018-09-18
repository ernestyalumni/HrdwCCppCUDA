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

cf. [`ctor.cpp`](https://github.com/ernestyalumni/HrdwCCppCUDA/blob/master/Cpp14/Strou/ctor_17/ctor.cpp)

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

## Constructors and Destructors

### Constructors and Invariants 

cf. pp. 484 17.2.1 Constructors and Invariants Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

*constructor* member with same name as its class; 
constructor declaration specifies an argument list (exactly as for a function), but has no return type.  
constructor's job is to initialize an object of its class

*class invariant* - something that must hold whenever a member function is called (from outside the class), and often constructor's initialization must establish this.  

class invariant stated as comments (often the case); constructor must make that true; e.g. 

``` 
Vector::Vector(int s)
{
	if (s < 0)
	{
		throw Bad_size{s};
	}
	sz = s;
	elem = new double[s];
}
``` 
This constructor tries to establish the invariant and if it can't, it throws an exception; no object is created and ctor must ensure no resources are leaked (Sec. 5.2, 13.3).  
Examples of resources are memory (Sec. 3.2.1.2), locks (Sec. 5.3.4), file handles (Sec. 13.3), thread handles (Sec. 5.3.1) 

Define an invariant to 
* focus design effort for the class (Sec. 2.4.3.2) 
* clarify behavior of the class (e.g. under error conditions, Sec. 13.2) 
* simplify the definition of member functions (Sec. 2.4.3.2, Sec. 16.3.1) 
* clarify class's management of resources (Sec. 13.3) 
* simplify documentation of class 

## Destructors

cf. pp. 485 17.2.2 Destructors and Resources Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   


A type that has no destructor declared, such as a built-in type, is considered to have a destructor that does nothing. 

### Base and Member Destructors

cf. pp. 486 17.2.3 Base and Member Destructors Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

ctors and dtors interact correctly with class hierarchies (Sec. 3.2.4, Ch. 20). ctor builds a class object "from the bottom up":

1. ctor invokes its base class ctors
2. then, it invokes member ctors
3. executes its own body.

dtors "tears down" object in reverse order:
1. dtor executes its own body,
2. then invokes its member dtors
3. invokes its base class dtors.

In particular a `virtual` base is constructed before any base might use it and destroyed after all such bases (Sec. 21.3.5.1). This ordering ensures a base or member isn't used before it has been initialized or used after it's been destroyed.

ctors execute member and base ctors in declaration order (**not the order of initializers**); if 2 ctors used a different order, dtor could not (without serious overhead) guarantee to destroy in reverse order of construction (See Sec. 17.4).

### `virtual` Destructors.

A destructor can be declared to be `virtual` and usually should be for a class with a virtual function. 

The reason we need a `virtual` destructor is that an object usually manipulated through the interface provided by a base class is often also `delete`d through that interface.

A common guideline is that a dtor for a base class must be either public and virtual or protected and nonvirtual. cf. http://www.gotw.ca/publications/mill18.htm

See https://en.cppreference.com/w/cpp/language/destructor and `./virtual_dtors_main.cpp`

cf. http://www.gotw.ca/publications/mill18.htm

1. Prefer to make interfaces nonvirtual, using Template Method, or NVI, Non-Virtual Interface Idiom; 
	- make interface of base class stable and nonvirtual, while delegating customizable work to nonpublic virtual functions responsible for implementing customizable behavior; virtual functions are designed to let derived classes customize behavior; it's better to not let publicly derived classes also customize inherited interface, which is supposed to be consistent.
2. Prefer to make virtual functions private
	- lets derived classes override function to customize behavior needed, without further exposing virtual functions directly by making them callable by derived classes; virtual functions exist to allow customization; unless they also need to be invoked directly from within derived classes' code, there's no need to make them anything but private
3. Only if derived classes need base implementation of virtual function, make virtual function protected.
4. base class destructor should be either public and virtual, or protected and nonvirtual. 
  - once execution reaches body of a base class dtor, any derived object parts have already been destroyed and no longer exist.

### Class Object Initialization 

cf. pp. 489 17.3 Class Object Initialization Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

We can initialize objects of a class for which we have not defined a ctor using 
* memberwise initialization
* copy initialization, or 
* default initialization (without an initializer or with an empty initializer list)

Where no ctor requiring arguments is declared, it's also possible to leave out the initializer completely. e.g. 

```
struct Work
{
	string author;
	string name;
	int year;
};

Work alpha;
```
For this, rules are not as clean; for statically allocated objects (Sec. 6.4.2), rules are exactly as if you had used `{}`, so value of `alpha` is `{"", "", 0}`; however, for local variables and free-store objects, default initialization is done only for members of class type, and members of built-in type are left uninitialized, so value of `beta` is `{"","",unknown}`.

Reason for this complication is to improve performance in rare critical cases. e.g.
```
struct Buf
{
	int count;
	char buf[16 * 1024];
};
```

You can use `Buf` as a local variable without initializing it before using it as a target for an input operation. Most local variable initializations aren't performance critical, and uninitialized local variables are a major source of errors.
If you want guaranteed initialization or simply dislike surprises, supply an initializer, such as `{}`. 

cf. pp. 491 17.3.2 Initialization Using Constructors. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

ctor often used to establish an invariant for its class, and to acquire necessary resources.

If a ctor is declared for a class, some ctor will be used for every object.

The usual overload resolution rules (Sec. 12.3) apply for ctors.

Note that `{}`-initializer notation doesn't allow narrowing (Sec. 2.2.2); another reason to prefer the `{}` style.

#### Default constructors

References and `const` must be initialized (Sec. 7.7, Sec. 7.5); therefore, a class containing such members can't be default constructed unless programmer supplies in-class member initializers (Sec. 17.4.4) or defines a default ctor that initializes them (Sec. 17.4.1). 

*When should a class have a default ctor?*
- "when you use it as the element type for an array, etc."

*For what types does it make sense to have a default value?*
*Does this type have a "special" value we can "naturally" use as a default?*

#### Initializer-List Constructors

cf. pp. 495 17.3.4 Initializer-List Constructors. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

[`std::initializer_list`](https://en.cppreference.com/w/cpp/utility/initializer_list)

##### `std::initializer_list`

Object of type `std::initializer_list<T>` is a lightweight proxy object that provides access to an array of objects of type `const T`. 

Initializer lists maybe implemented as a pair of pointers or pointer and length; copying a `std::initializer_list` doesn't copy underlying objects.

(since C++14)
Underlying array is temporary array of type `const T[N]`, which each element is copy-initialized (except narrowing conversions invalid) from corresponding element of original initializer list. 
	Lifetime of underlying array is same as any temporary object.

See `../stdinitializer_list_eg.cpp`

Distinction between direct initialization and copy initialization (Sec. 16.2.6) is maintained for `{}` initialization. 

For a container, this implies that distinction is applied to both container and its elements:
* container's initializer-list ctor can be `explicit` or not
* ctor of element type of the initializer list can be `explicit` or not.

#### Member and Base Initialization

cf. pp. 501 17.4.1 Member Initialization. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

It's usually a good idea to be explicit about initializing members. Note that an "implicitly initialized" member of a built-in type is left uninitialized (Sec. 17.3.1)

A ctor can initialize members and bases of its class, but not members or bases of its members or bases. e.g.

```
struct B
{
	B(int);
};

struct BB : B {};

struct BBB: BB {
	BBB(int i) : B(i) {} // error: trying to initialize base's base
}
```

*delegating constructor (or forwarding constructor)* - member-style initializer using class's own name (its ctor name) calls another ctor as part of the construction.

```
class X
{
		int a;
	public:
		X(int x) { if (0 < x && x <= max) a = x; else throw Bad_X(x);}
		X() : X{42}{}
		X(string s): X {to<int>(s)} {}
}
```

You can't both delegate and explicitly initialize a member.
Delegating by calling another ctor in a ctor's member and base initializer list is very different from explicitly calling a ctor in the body of a ctor (simply creates a new unnamed object (a temporary) and does nothing).

An object is not considered constructed until its ctor completes. A dtor won't be called for an object unless its original ctor completed.


Member initialization is done in declaration order (Sec. 17.2.3), so first `m1` is initialized to value of a global variable `count2`. 

cf. `member_base_initialization_main.cpp`

##### `static` Member Initialization

`static` class member is statically allocated rather than part of each object of the class.  
Generally, `static` member declaration acts as a declaration for a definition outside the class.
However, for a few simple special cases, it's possible to initialize `static` member in class declaration. `static` member must be `const` of an integral or enumeration type, or `constexpr` of literal type. 

If (and only if) you use an initialized member in a way that requires it to be stored as an object in memory, member must be (uniquely) defined somewhere. Initializer may not be repeated:

```
const int Curious::c1; 	// don't repeat initializer here
const int* p = &Curious::c1; // OK: Curious::c1 has been defined
```

Main use of member constants is to provide symbolic names for constants needed elsewhere in class declaration.

## Copy and Move.

cf. pp. 507 17.5 Copy and Move. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

* *Copy* `x=y` effect is that values `x` and `y` are both equal to `y`'s value before assignment. `x=y` should imply `x==y`
* *Move* leaves `x` with `y`'s former value and `y` with some *moved-from state*.

Typically, a move can't throw, whereas copy might (because it may need to acquire a resource), and move is often more efficient than a copy. 
When you write a move operation, you should leave source object in a valid but unspecified state because it'll eventually be destroyed and dtor can't destroy an object left in invalid state.
Als, standard-library algorithms rely on being able to assign to (using move or copy) a moved-from object.
So, design your moves not to throw, and leave their source object in state that allows destruction and assignment.

cf. pp. 509 17.5.1.1 Beware of Default Constructors. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

When writing a copy operation, be sure to copy every base and member. For larger classes, chances of forgetting go up; worse when someone long after the initial design adds a member to a class, it's easy to forget to add it to the list of members to be copied. This is 1 reason to prefer the default (compiler-generated) copy operations (Sec. 17.6).

cf. pp. 509 17.5.1.2 Copy of Bases. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.



A `virtual` base (Sec. 21.3.5) may appear as base of several classes in a hierarchy.
A default copy ctor (Sec. 17.6) will correctly copy it.
If you define your own copy ctor, simplest technique is to repeatedly copy the `virtual` base. (?)
	Where base object is small, and `virtual` base occurs only a few times in a hierarchy, that can be more efficient than techniques for avoiding the replicated copies. (???)

*shallow copy* - e.g. object that contain pointer, default copy operation copies a pointer member, but doesn't copy the object (if any) that it points to: e.g.
```
struct S
{
	int* p;
};

S x {new int{0}};
S y {x};
```
we can manipulate part of `x`'s state through `y`.
Shallow copy leaves 2 objects with a *shared state*.

*deep copy* - copying complete state of an object

Often, better alternative to deep copy is not a shallow copy, but a move operation, which minimizes copying without adding complexity (Sec. 3.3.2, Sec. 17.5.2)


cf. pp. 512 17.5.1.3 Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

### *copy-on-write* idiom

cf. pp. 512 17.5.1.3 Meaning of Copy. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Copy-on-write

Idea is that a copy doesn't actually need independence until a shared state is written to, so we can delay the copying of the shared state until just before the first write to it.

cf. `./CopyOnWrite/*`










cf. pp. 512 17.5.1.4 Slicing. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.









### Move

Many objects in a computer resemble physical objects (which we don't copy without need and only at considerable cost) more than integer values (which we typically copy because that's easier and cheaper than alternatives).
	Examples are locks, sockets, file handles, threads, long strings, and large vectors.

Move ctors and move assignments take non-`const` (rvalue) reference arguments: they can, and usually do, write to their argument. However, argument of a move operation must always be left in a state that the dtor can cope with (and preferably deal with very cheaply and easily).

For resource handles, move operations tend to be significantly simpler and more efficient than copy operations.
  	In particular, move operations typically don't throw exceptions; they don't acquire resources or do complicated operations, so they don't need to.

How does compiler know when it can use a move operation rather than a copy operation?
In a few cases, such as for return value, language rules say it can (because next action is defined to destroy the element)
In general, we have to tell it by giving an rvalue reference argument. 

`std::move` is a standard-library function return an rvalue reference to its argument (Sec. 35.5.1): `std::move(x)` means "give me an rvalue reference to `x`". 

#### Default operations; Generating default operations.

cf. pp. 517 17.5.2 Generating Default Operations. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

Compiler can generate copy and dtor for us as needed. By default, a class provides
* default ctor
* copy ctor
* copy assignment
* move ctor
* move assignment
* dtor

By default, compiler generates each of these operations if a program uses it. However,
- If programmer declares any ctor, default ctor not generated.
- If programmer declares a copy operation, a move operation, or a destructor for a class, no copy operation, move operation, or destructor is generated for that class.

Using `= default` is always better than writing your own implementation of the default semantics.

Default menaing of each generated operation, as implemented when compiler generates it, is to apply the operation to each base and non-`static` data member of the class; i.e. we get memberwise copy, memberwise default construction, etc.

Note that value of a moved-from object of a built-in type is unchanged. That's the simplest and fastest thing for the compiler to do.

Whenever possible,
1. Establish an invariant in a ctor (including possibly resource acquisition)
2. Maintain invariant with copy and move operations
3. Do any needed cleanup in dtor

For every class, we should ask:
1. Is a default ctor needed (because default 1 is not adequate or has been suppressed by another ctor)?
2. Is dtor needed (e.g. because some resource needs to be released)?
3. Are copy operations needed (because default copy semantics is not adequate, e.g. because class ismeant to be a base class or because it contains pointers to objects that must be deleted by the class)?
4. Are move operations needed (because default semantics is not adequate, e.g. because an empty object doesn't make sense)?

#### `delete`d Functions

cf. pp. 524 17.6.4 `delete`d Functions. Ch. 17 *Construction, Cleanup, Copy, and Move*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

Most obvious use is to eliminate otherwise defaulted functions. For example, it's common to want to prevent copying of classes used as bases because such copying easily leads to slicing (Sec. 17.5.1.4):

We can `delete` any function that we can declare.
e.g. eliminate a specialization from set of possible specializations of a function template.

- eliminate an undesired conversion. e.g.
```
struct Z
{
	Z(double); // can initialize with a double
	Z(int) = delete; // but not with an integer
}

void f()
{
//	Z z1 {1}; // error: Z(int) deleted
	Z z2 {1.0}; // OK
}
```

You can't have a local variable that can't be destroyed (Sec. 17.2.2),
you can't allocate object on free store when you have `=delete`d its class's memory allocation operator (Sec. 19.2.5).

Note difference between a `= delete`d function and 1 that simply hasn't been declared.
In former, compiler notes that programmer has tried to use the `delete`d function and gives an error. 
In latter, compiler looks for alternatives, such as not invoking a dtor, or using a global `operator new()`.










