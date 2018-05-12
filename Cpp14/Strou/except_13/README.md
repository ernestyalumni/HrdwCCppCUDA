# Exception Handling; `try`, `catch`

cf. pp. 343, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

cf. pp. 343, Sec. 13.1 Error Handling, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

General notes; "for effective error handling, the language mechanisms must be used based on a strategy." 
  - *exception-safety guarantees* 
  - *Resource Acquisition Is Initialization (RAII)* 
Both exception-safety guarantees and RAII depend on specification of *invariants*.  

## Exceptions

cf. pp. 344, 13.1.1 Exceptions Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

*exception* - help get information from point where error is detected to a point where it can be handled.  
- function that can't cope with a problem *throws* an exception, hoping that its (direct or indirect) caller can handle the problem. 
- function that wants to handle a kind of problem indicates that by *catch*ing corresponding exception 

A called function can't just return with an indication that an error happened; 
- if program is to continue working (and not just print an error message and terminate), returning function must leave program in a good state and not leak any resources.

An exception is an object `throw` to represent occurrence of an error.  
* It can be any type that can be copied, but it's strongly recommended to use only user-defined types specifically defined for that purpose. 

cf. pp. 348, 13.1.4.2 Exceptions That Are Not Errors, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Think of an exception as meaning "some part of the system couldn't do what it was asked to do"

Exception `throw`s should be infrequent compared to function calls or structure of the system.

When at all possible, stick to the "exception handling is error handling" view. When this is done, code is clearly separated into 2 categories: ordinary code and error-handling code. 

cf. pp. 349, 13.1.5 When You Can't Use Exceptions, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

"Use of exceptions is the only fully general and systematic way of dealing with errors in a C++ program." 

cf. pp. 351, 13.1.7 Exceptions and Efficiency, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

In principle, exception handling can be implemented so there's no run-time overhead when no exception is thrown. 

cf. pp. 353, 13.2 Exception Guarantees, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

To recover from an error, i.e. catch an exception and continue executing a program, we need to know what can be assuemd about state of the program before and after the attempted recovery action.  
Call an operation *exception-safe* if operation leaves program in a valid state when operation is terminated by throwing an exception. 

When reasoning about objects, assume class has class invariant (cf. Sec. 2.4.3.2, Sec. 17.2.1). Assume this invariant is established by its constructor and maintained by all functions with access to object's representation until object's destroyed.  
*valid state* - constructor has completed and destructor has not yet been entered. 

Before a `throw`, a function must place all constructed objects in valid states. However, such a valid state may be 1 that doesn't suit the caller.  
e.g. a `string` may be left as the empty string or a container may be left unsorted. 
Thus, for complete recovery, an error handler may have to produce values that are more appropriate/desirable for application than the (valid) ones existing at the entry to a `catch`-clause. 

C++ standard library provides a generally useful conceptual framework for design for exception-safe program components, i.e. 
C++ standard library - follow it as an example for exception-safety. 
  * *basic guarantee* for all operations: basic invariants of all objects are maintained; no resources, such as memory are leaked. Particularly, basic invariants of every built-in and standard-library type guarantee that you can destroy an object or assign to it after every standard-library operation (iso.17.6.3.1).  
  * *strong guarantee* for key operations: either operation succeeds, or *it has not effect.* This guarantee is provided for key operations, such as `push_back()`, single-element `insert()` on a `list`, and `uninitialized_copy()`. 
  * *nothrow guarantee* for some operations: in addition to provided the basic guarantee, some operations are guaranteed not to throw an exception. This guarantee is provided for a few simple operations, such as `swap()` of 2 containers and `pop_back()` 

  Both basic guarantee and strong guarantee are provided on the condition that 
  * user-supplied operations (such as assignments and `swap()` functions) do not leave container elements in invalid states 
  * user-supplied operations don't leak resources, and 
  * destructors don't throw exceptions (iso.17.6.5.12)

Violating a std lib requirement, such as having a destructor exit by throwing an exception, is logically equivalent to violating a fundamental language rule, such as dereferencing a null pointer. 

In particular, operation that throws an exception must not only leave its operands in well-defined states, but must also ensure that every resource it acquired is (eventually) released. 
e.g. at the point where an exception is thrown, all memory allocated must be either deallocated or owned by some object, which in turn must ensure that memory is properly deallocated. 

cf. `ThrowRequireResource.cpp` 

Consider `gdb`ing `./ThrowRequireResource` (the executable, that is), and obtaining 
``` 
terminate called after throwing an instance of 'int'

Program received signal SIGABRT, Aborted.
0x00007ffff76c0428 in __GI_raise (sig=sig@entry=6)
    at ../sysdeps/unix/sysv/linux/raise.c:54
``` 

In general, we must assume that very function that can throw an exception will throw one. 

"Resource Acquisition Is Initialization" (RAII; Sec. 5.2) - technique for managing resources using local objects; general technique that relies on the properties of constructors and destructors and their interaction with exception handling. 

A constructor (ctor) tries to ensure that its object is completely and correctly constructed. When that can't be achieved, a well-written ctor restores - as far as possible - state of the system to what it was before creation (attempt?).  
Ideally, a well-designed ctor always achieves 1 of these alternatives and doesn't leave its object in some "half-constructed" state. 

Consider class `X` for which a ctor needs to acquire 2 resources; a file `x` and mutex `y` (Sec. 5.3.4).   
Use objects of 2 classes, `File_ptr` and `std::unique_lock` (Sec. 5.3.4) to represent acquired resources.  
The acquisition of a resource is represented by initialization of local object that represents the resource: 
``` 
class Locked_file_handle 
{
    File_ptr p;
    unique_lock<mutex> lck;
  public:
    X(const char* file, mutex& m):
      p{file, 'rw'},  // acquire "file"
      lck{m}          // acquire "m"
    {}
    //...
}
```  

finally

cf. `Finally.cpp` 

## Enforcing Invariants 

cf. pp. 359, 13.4 Enforcing Invariants, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

When a ctor can't establish its class invariant (Sec. 2.4.3.2, Sec. 17.2.1), object isn't usable. 
- Typically, throw exceptions.

However, there are programs for which throwing an exception isn't an option (Sec. 13.1.5), 
* *Just don't do it*
* *Terminate program* Violating a precondition is a serious design error 

All 3 shared a common view that preconditions should be defined and obeyed.

need to *assert*: 
* choose between compile-time asserts (evaluated by compiler) and run-time asserts (evaluated at run time) 
* for run-time asserts, need  a choice of throw, terminate, or ignore
* no code shoudl be generated unless some logical condition is `true` , e.g. some runtime asserts shouldn't be evaluated unless logical condition is `true`. Usually, logical condition is something like a debug flag, level of checking, or mask to select aong asserts to enforce 
* asserts shouldn't be verbose or complicated to write (because they can be very common) 

The standard offers 2 simple mechanisms: 
* `<cassert>` 
* `static_assert(A, message)`, which unconditionally checks its assertion `A` at compile time (Sec.2.4.3.3). If assertion fails, compiler writes out `message` and compilation fails. 

destructors should not throw, so don't use a throwing `Assert()` is a destructor 

## Throwing and Catching Exceptions 

cf. pp. 363, 13.5 Throwing and Catching Exceptions, Ch. 13 **Exception Handling** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

There's a small standard-library hierarchy of exception types (SEc. 13.5.2) that can be used either directly or as base classes. e.g.
``` 
struct My_error2: std::runtime_error 
{
  const char* what() const noexcept
  {
    return "My_error2";
  }
};
``` 


