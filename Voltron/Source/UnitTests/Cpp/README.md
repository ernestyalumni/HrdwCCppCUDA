cf. Bjarne Stroustrup. The C++ Programming Language, 4th Edition. Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842

# Pointers, Arrays, References

cf. Stroustrup (2013), pp. 171, Ch. 7 Pointers, Arrays, and References

cf. 7.1 Introduction, pp. 171, Stroustrup (2013)

In C++ (most) objects "have identity" - i.e. reside at a specific address in memory, and
- object can be accessed if you know its address and its type.

The language constructs for holding and using addresses are *pointers and references.*

## Pointers

cf. 7.2 Pointers, pp. 172, Stroustrup (2013)

For type `T`, `T*` is type "pointer to `T`"
- variable of type `T*` can hold the address of an object of type `T`

Pointers (its implementation, specifically, is intended to) map directly to addresses of the machine on which program runs. 

cf. 6.3.1 pp. 154, Stroustrup (2013).

Structure of Declarations, 3rd part:
* A declarator optionally including a name (e.g. `p[7], n`, or `*(*)[]`)

Declarator is composed of (an optional) name, and optionally some declarator operators.
Most common declarator operators are:

Declarator Operators


| Declarator Operators   |
| :------------ |:------------  | :----|
| prefix      | `*` | pointer |
| prefix      | `*const` | constant pointer |
| prefix      | `*volatile` | volatile pointer |
| prefix      | `&` | lvalue reference (Sec. 7.7.1) |
| prefix      | `&&` | rvalue reference (Sec. 7.7.2) |
| prefix      | auto | function (using suffix return type) |
| postfix      | `[]` | array |
| postfix      | `()` | function |
| postfix      | `->` | returns from function |

Their use would be simple if they were all either prefix or postfix (!!!).
However, `*`, `[]`, `()` were designed to mirror their use in expressions (Sec. 10.3).
Thus, `*` is prefix, `[]` `()` are postfix.
Postfix declarator operators bind tighter than prefix ones.
Consequently, `char* kings[]` is an array of pointers to `char`, whereas `char(*kings)[]` is a pointer to an array of `char`. 

Pointers to functions Sec. 12.5

Pointers to class members in Sec. 20.6

### `void*`

In low-level, we occasionally need to store or pass along address of memory location without actually knowing what type of object stored there (e.g. Linux System Programming).

`void*` - "pointer to an object of unknown type"

A pointer to any type of object can be assigned to variable of type `void*`, but pointer to function (Sec. 12.5), or pointer to member (Sec. 20.6) cannot. 

In addition
- `void*` can be assigned to another `void*`
- `void*`s can be compared for equality and inequality, and
- `void*` can be explicitly converted to another type.

Other operations unsafe because compiler cannot know what kind of object is really pointed to.

To use `void*`, we must explicitly convert it to a ptr to specific type.

Primary use for `void*` is for passing pointers to functions that aren't allowed to make assumptions about type of object and for returning untyped objects from functions (e.g. Linux System Programming).

Where used for optimization, `void*` can be hidden behind type-safe interface (Sec. 27.3.1).

### `nullptr` 

cf. 7.2.2. pp. 173-174

literal `nullptr` represents null pointer, i.e. a pointer that doesn't point to an object. 
- Can be assigned to any pointer type, but not to other built-in types.


## Arrays

cf. Stroustrup (2013), pp. 174, 7.3 Arrays

If what you want is a simple fixed-length sequence of objects of a given type in memory, an array is the ideal solution.

There's no array assignment.
Name of an array implicitly converts to pointer to its first element.

Avoid arrays in interfaces (e.g. as function arguments) because implicit conversion to pointer is root cause of common errors in C, C-style C++ code.

Most widely used kinds of arrays is a zero-terminated array of `char`; that's the way C stores strings.
- Often, `char*` or `const char*` assumed to point to zero-terminated sequence of characters.

## String Literals

A *string literal* is a character sequence enclosed within double quotes:
```
"this is a string"
```
A string literal contains 1 more character than it appears to have; it's terminated by null character, `\0`, with value `0`. e.g.
```
sizeof("Bohr") == 5;
```

A string literal is statically allocated, so that it's safe to return 1 from a function. e.g.
```
const char* error_message(int i)
{
  // ...
  return "range error";
}
```

### Larger Character Sets

string with prefix `L`, e.g. `L"angst"` is string of wide characters; its type is `const wchar_t[]`

TODO: take notes on Sec. 7.3.2.2, pp. 178, Stroustrup (2013)


## Pointers into Arrays

Implicit conversion of an array name to a pointer to the initial element of the array is extensively used in function calls in C-style code.

There is no implicit or explicit conversion from a pointer to an array.

Implicit conversion of array argument to pointer means size of array is lost to called function.
- To determine size, C standard-library functions taking pointers to characters, `strlen()` relies on zero to indicate end-of-string

## Navigating Arrays: Pointer Arithmetic

cf. 7.4.1 Navigating Arrays, pp. 181, Stroustrup (2013)

Efficient and elegant access to arrays is key to many algorithms (see Sec. 4.5, Ch. 32). 
- Access can be achieved either through a pointer to an array plus index or through pointer to an element.
  * No inherent reason why 1 version should be faster than other

Subscripting a built-in array is defined in terms of the pointer operations `+` and `*`. 


Complicated pointer arithmetic is usually unnecessary and best avoided. 

Arrays *are not self-describing* because number of elements of an array isn't guaranteed to be stored with the array. 
- This implies that to traverse asn array that doesn't contain a terminator the way C-style strings do, we must supply the number of elements.

## Multidimensional Arrays



cf. Functional Programming in C++. Ivan Čukić. November 2018 ISBN 9781617293818 
320 pages Manning Publications


# Function objects, Ch. 3, pp. 45

See `Functions_tests.cpp`.

## Call operator overloading

pp. 50, 3.1.3 Call operator overloading implies that *classes with overloaded call operators* are better than function pointers.

### Lambda syntax, pp. 55, 3.2.1

```
[a, &b](int x, int y) { return a * x + b * y; }
```

`[a, &b]` is the head, `(int x, int y)` is the argument, `{ return a * x + b * y}` is the body.

The head of the lambda specifies which variables from surrounding scope will be visible inside lambda body.


* `[a, &b]` - `a` is captured by value, `b` captured by reference
* `[]` lambda doesn't use any variable from surrounding scope.
  - These lambdas don't have any internal state and can implicitly be cast to ordinary function pointers.
* `[&]` captures all variables used in lambda body by reference.
* `[=]` captures all variables used in lambda body by value
* `[this]` captures `this` pointer by value
* `[&, a]` captures all variables by reference, except `a` which is captured by value
* `[=, &b]` captures all variables by value, except `b` which is captured by reference.

wildcards `[&]`, `[=]`
- prefer explicitly enlisting all variables needed in lambda body

### Creating arbitrary member variables in lambdas, 3.2.3, pp. 59

## Currying, 4.2

### Currying and partial function application, 4.2.3.

Currying con: currying is more limiting in the sense that it must bind arguments in order:
- first argument first, last argument last

`std::bind` advantage: you can bind any of the arguments, whereas the curried function first binds the first argument.

`std::bind` drawback - you need to know exactly how many arguments the function you're passing to `std::bind` has. You need to bind each argument to either a value (or a variable, a reference) or placeholder. 
- with curried function, you don't need to care about that; you define value for first function argument, query returns function that accepts all other arguments, no matter how many there are.

# Purity: Avoiding mutable state, Ch. 5

## Pure functions and referential transparency

cf. 5.2 Pure functions and referential transparency. pp. 103

Design flaw of class data encapsulation: having several components in the software system be responsible for the same data, without knowing when another component is changing that data. The *simplest* way to fix this is to forbid changing any data.

Instead of saying you can't change state, let's see how to design software in a way that keeps mutations and side effects to a minimum.

Expression is *referentially transparent* if the program wouldn't behave any differently if we replaced the entire expression with just its return value.

As soon as a function call can't be completely replaced by its return value without changing the behavior of the program, it has *observable side effects.*

cf. pp. 106, 5.3 Programming without side effects

In pure functional programming, instead of changing a value, you create a new one. Instead of changing a property of an object, you create copy of that object, and just the value of that property is changed to the new value.

cf. 5.4 Mutable and immutable state in a concurrent environment, pp. 111

When `mutex` is necessary: when you *want* to have things running in parallel - whether for efficiency or something else; mutexes solve the problem with concurrency by removing concurrency.

From -David Butenhof on comp.programming.threads, (mutex - which stands for mutual exclusion), acts like a "bottleneck".

Mutexes, like `for` loops and recursion, are low-level constructs useful for implementing higher-level abstractions for concurrent programming, but use it sparingly.
- problems appear only if you have mutable data shared across different concurrently running processes. Solutions:
1. not have concurrency
2. not to use mutable data
3. have mutable data, not share it.

pp. 113 
Mutable not shared ok
Immutable shared and Immutable Not shared ok 

Mutable shared not ok.

## Logical and internal `const`-ness

cf. 5.5.1 Logical and internal const-ness

Instead of making all member variables constant, make all (public) member functions constant:

```
class Person
{
  public:
    std::string name() const;
    std::string surname() const;
  private:
    std::string name_;
    std::string surname_;
};
```
provides both *logical* `const`-*ness* (user-visible data in object never changes)
and
*internal* `const`-*ness* (no changes to internal data of object)

pp. 117 
- create `mutable` member variable in your class, member variable that can be changed even from const member functions; because you want to guarantee that class is immutable from user's perspective, you have to ensure 2 concurrent calls to `employmnet_history` can't return different values.
  * **pattern** of implementing classes that are immutable on outside, but sometimes need to change internal data.
    - it's required for constant member functions that either class data is kept unchanged or all changes are synchronized (unnoticeable even in the case of concurrent invocations).

# Lazy Evaluation, Ch. 6

See `Utilities/LazyValuation.h`

# Algebraic data types, Ch. 9, pp. 174

In functional world, building new types from old ones usually done with 2 operations:
* sum
* product

These new types are thus called *algebraic*.

Product of 2 types `A`, `B` is a new type that contains an instnace of `A`, instance of `B` (it'll be Cartesian product of set of all values of type `A`, and set of all values of `B`)

e.g. `std::pair`, `std::tuple`

Sum types: sume type of types A and B is a type that can hold an instance of A or an instance of B, but not both at the same time.

Enums are a special kind of sum type; enum is a sum type of one-element sets; define enum specifying different values it can hold, instance of that enum type can hold exactly 1 of those values, and treat these values as 1-element sets.

sums are disjoint union or coproduct
https://en.wikipedia.org/wiki/Coproduct

## Sum types through inheritance

cf. 9.1.1. Sum types through inheritance, Cukic, pp. 177



# Template metaprogramming, Ch. 11, pp. 226

