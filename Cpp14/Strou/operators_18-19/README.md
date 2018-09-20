# Operators; operator Overloading, special operators

cf. pp. 527, Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

## Operator Functions

Functions defining meanings for the following operators (Sec. 10.3) can be declared:

```
+ - * / % - &
| ~ ! = < > += 
-= *= /= %= ^= &= !=
<< >> >>= <<= == != <=
>= && || ++ -- ->* ,
-> [] () new new[] delete delete[]
```

Following operators can't be defined by a user:
`::` scope resolution (Sec. 6.3.4, Sec. 16.2.12)
`.` member selection (Sec. 8.2)
`.*` member selection through pointer to member (Sec. 20.6)
They take a name, rather than a value, as their second operand and provide primary means of referring to members.

The named "operators" can't be overloaded because they report fundamnetal facts about their operands:
`siezof` size of object (Sec. 6.2.8)
`alignof` alignment of object (Sec. 6.2.9)
`typeid` `type_info` of an object (Sec. 22.5)

Finally, ternary conditional expression operator can't be overloaded
`?:` conditional evaluation (Sec. 9.4.1)

user-defined literals (Sec. 19.2.6) defined by using `operator""` notation.
`operator T()` defines conversion to type `T` (Sec. 18.4)

Name of an operator function is keyword `operator` followed by operator itself, for example, `operator<<`. 

A use of the operator is only a hosrthand for an explicit call of the operator function, e.g.
```
void f(complex a, complex b)
{
  complex c = a + b; // shorthand
  complex d = a.operator+(b); // explicit call
}
```

Overload resolution (Sec. 12.3) determines which, if any, interpretation is used.

### Predefined Meanings for Operators

cf. pp. 531 18.2.2 Predefined Meanings for Operators. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

operators `=` (assignment), `&` (address-of) and `,` (sequencing; Sec. 10.3.2) have predefined meanings when applied to class objects; can be eliminated ("deleted", Sec. 17.6.4):

```
class X
{
  public:

    void operator=(const X&) = delete;
    void operator&() = delete;
    void operator,(const X&) = delete;
};
```

cf. pp. 532 18.2.3 Operators and User-Defined Types. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

In particular, it's not possible to define an operator function that operates exclusively on pointers. This ensures that C++ is extensible but not mutable (with the exception of operators `=`, `&`, and `,` for class objects)

Enumerations are user-defined types so that we can define operators for them, e.g.
```
enum Day 
{
  sun, mon, tue, wed, thu, fri, sat
};

Day& operator++(Day& d)
{
  return d = (sat == d) ? sun : static_cast<Day>(d+1);
}
```

### Passing Objects.

cf. pp. 532 18.2.4 Passing Objects. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

We have limited choices of how to pass arguments to the operator function and how it returns its value; e.g. we can't require pointer arguments and expect programmers to use address-of operator or return a pointer and expect user to dereference it: `*a = &b + &c` isn't acceptable.

For arguments, we have 2 main choices (Sec. 12.2)
* pass-by-value
* pass-by-reference

For small objects, say, 1 to 4 words, call-by-value is typically a viable alternative and often 1 that gives the best performance.
However, performance of argument passing and use depends on machine architecture, compiler interface conventions (Application Binary Interfaces; ABIs), and number of times an argument is accessed (it's almost always faster to access an argument passed by value than 1 passed by reference).

Larger objects, pass by reference, e.g. `Matrix` (Sec. 17.5.1) is most likely larger than a few words.

In particular, use `const` references to pass large objects that aren't meant to be modified by the called function (Sec. 12.2.1).

Typically, an operator returns a result.
Returning a ptr or reference to a newly created object is usually a very bad idea: using a ptr gives notational problems, 
referring to an object on the free store (whether by pointer or by reference) results in memory management problems.
Instead, return objects by value.
For large objects, such as a `Matrix`, define move operations to make such transfers of values efficient (Sec. 3.3.2, Sec. 17.5.2).

```
Matrix operator+(const Matrix& a, const Matrix& b) // return-by-value
```

cf. `../ctor_17/Matrix/*`

Note that operators that return 1 of their argument objects can - and usually do - return a reference. e.g. 
```
Matrix& Matrix::operator+=(const Matrix& a)
```
This is particularly common for operator functions that are implemented as members.

If a function simply passes an object to another function, an rvalue reference argument should be used (Sec. 17.4.3, Sec. 23.5.2.1, Sec. 28.6.3)

cf. pp. 534 18.2.5 Operators in Namespace. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Consider a binary operator `@`. If `x` is of type `X` and `y` is of type `Y`, `x@y` resolved like this:
* If `X` is a class, look for `operator@` as a member of `X` or as member of a base of `X`, and 
* look for declarations of `operator@` in context surrounding `x@y`; and 
* if `X` is defined in namespace `N`, look for declarations of `operator@` in `N`; and
* if `Y` is defined in namespace `M`, look for declarations of `operator@` in `M`

Declarations for several `operator@`s may be found and overload resolution rules (Sec. 12.3) are used to find best match, if any.
This lookup mechanism applied only if operator has at least 1 operand of a user-defined type.
Therefore, user-defined conversions (Sec. 18.3.2, 18.4) will be considered.
Note that type alias is just a synonym and not a separate user-defined type (Sec. 6.5).
Unary operators resolved analogously.
Note that in operator lookup no preference given to members over nonmembers.
This differs from lookup of named functions (Sec. 14.2.4). 
Lack of hiding of operators ensures built-in operators are never inaccessible and that users can supply new meanings for an operator without modifying existing class declarations.

### Conversions.


cf. pp. 539 18.3.3 Conversions of Operands. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

3 version of each of the 4 standard arithmetic operators, e.g.
```
complex operator+(complex, complex);
complex operator+(complex, double);
complex operator+(double, complex);
```

Alternative to providing different versions of a function for each combination of arguments is to rely on conversions. e.g. we could simply declare 1 version of the equality operator for `complex`

```
bool operator==(complex, complex);
```

Reasons for preferring to define separate functions: conversion could impose overhead in some cases, or simpler algorithm used to specific argument types;
Otherwise, relying on conversions and providing only the most general variant of a function - plus possibly a few critical variants - contain the combinatorial explosion of variants that can arise from mixed-mode arithmetic.

Where several variants of a function or operator exist, compiler must pick "the right" variant based on argument types and available (standard and user-defined) conversions.

### Literals

cf. pp. 540 18.3.4 Literals. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

We have literals of built-in types.

For `complex` we can declare ctors `constexpr` (Sec. 10.4)

Introduce user-defined literal (Sec. 19.2.6). e.g. in particular, we could define `i` to be a suffix meaning "imaginary." e.g.
```
constexpr complex<double> operator ""i(long double d) // imaginary literal
{
  return {0, d}; // complex is a literal type
}
```

## Type Conversion

Type conversion can be accomplished by
* ctor taking a single argument (Sec. 16.2.5)
* conversion operator (Sec. 18.4.1)









