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

In either case, conversion can be 
* `explicit`, i.e. conversion is only performed in a direct initialization (Sec. 16.2.6), i.e. as an initializer not using a `=`
* implicit, i.e. applied wherever it can be used unambiguously (Sec. 18.4.3), e.g. as a function argument.

### Conversion operators

cf. pp. 543 18.4.1 Conversion Operators. Ch. 18 *Operator Overloading*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.   

Using a ctor taking a single argument to specify type conversion is convenient, but a ctor can't specify 
1. an implicit conversion from a user-defined type to a built-in type (because built-in types aren't classes), or 
2. conversion from new class to previously defined class (without modifying declaration for the old class)

These problems can be handled by defining a 
*conversion operator* - a member function 

```
X::operator T()
```
where `T` is a type name, defines conversion from `X` to `T`. 

implicit conversion from `X` to `T`; note that type being converted to is part of the name of the operator and can't be repeated as the return value of the conversion function:

```
X::operator T() const { return v; } // right
T X::operator T() const { return v; } // error
```
in this respect, a conversion operator resembles a ctor.

Whenever a `X` appears where a `T` is needed, appropriate `T` is used.

Conversion functions appear to be useful for handling data structures when reading (implemented by a conversion operator) is trivial.

`istream` and `ostream` types rely on a conversion function to enable statements such as:
```
while (cin >> x)
  cout << x;
```

However, it's typically *not* a good idea to define implicit conversion from 1 type to another such that information is lost in conversion.

In general, it's wise to be sparing in the introduction of conversion operators.

Probably the best idea is initially to do conversions by named functions, such as `X::make_int()`. 

### `explicit` Conversion Operators

Conversion operators tend to be defined so that they can be used everywhere; however, it's possible to declare a conversion operator `explicit`, and have it apply only for direct initialization (Sec. 16.2.6), where an equivalent `explicit` ctor would've been used, e.g.

```
explicit operator bool() const noexcept;
```
Reason to declare conversion operator `explicit` is to avoid its use in surprising contexts.

Only 1 level of user-defined implicit conversion is legal.
In some cases, a value of desired type can be constructed in more than 1 way; such cases are illegal.

# Special Operators

cf. pp. 549 Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

## Subscripting, `operator[]`

cf. pp. 550 Sec. 19.2.1 Subscripting. Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

`map`, `unordered_map`

## Function Call

cf. pp. 550 Sec. 19.2.2 Function Call. Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

Function call, that is, notation *expression(expression-list)* can be interpreted as a binary operation, with *expression* as left-hand operand, and *expression-list* as right-hand operand. 
The call operator `()` can be overloaded in the same way as other operators can.
e.g.

```
struct Action
{
  int operator()(int);
  pair<int, int> operator()(int, int);
  double operator()(double);
};
```
Argument list for `operator()()` is evaluated and checked according to usual argument-passing rules. 
Overloading function call operator seems to be useful primarily for defining types that have only a single operation and for types for which 1 operation is predominant.

The most obvious and most important use of `()` operator is to provide usual function call syntax for objects that in some way behave like functions.
Such function objects allow us to write code that takes nontrivial operations as parameters. 

In many cases, it's essential function objects can hold data needed to perform their operation. e.g. define a class with an `operator()()` that adds stored value to its argument:

(EY: functor?)
```
class Add
{
  public:
    Add(complex c):
      val_{c} // save a value
    {}

    Add(double r, double i):
      val_{{r, i}} 
    {}

    void operator()(complex& c) cosnst
    {
      c+= val; // add a value to argument
    }

  private:
    complex val_;
}
```
An object of class `Add` is initialized with a complex number, and when invoked using `()`, it adds that number to its argument. e.g.


```
void h(vector<complex>& vec, list<complex>& lst, complex z)
{
  for_each(vec.begin(), vec.end(), Add{2, 3});
  for_each(lst.begin(), lst.end(), Add{z});
}
```
This all works because `for_each` is a template that applies `()` to its third argument without caring exactly what that 3rd argument really is: cf. Sec. 3.4.3, Sec. 33.4

Note that lambda expression (Sec. 3.4.3, Sec. 11.4) is basically a syntax for defining a function object. e.g.

```
void h2(vector<complex>& vec, list<complex>& lst, complex z)
{
  for_each(vec.begin(), vec.end(), [](complex& a) { a+= {2, 3}; });
  for_each(lst.begin(), lst.end(), [](complex& a) { a += z});
}
```
In this case, each of the lambda expressions generates the equivalent of the function object `Add`.

Other popular uses of `operator()()` are as a substring operator and as subscripting operator for multidimensional arrays (Sec. 29.2.2, Sec. 40.5.2)

`operator()()` must be a non-`static` member function.

Function call operators are often templates (Sec. 29.2.2, Sec. 33.5.3)

## Dereferencing `X* operator->()`

dereferencing operator `->` (also known as the *arrow* operator) is a unary postfix operator, `operator->()` doesn't depend on the member `m` pointer to. However, there's no new syntax introduced, so a member name is still required after the `->`, e.g. `X* q2 = p.operator->();`.

For ordinary ptrs, use of `->` is synonymous with some uses of unary `*` and `[]`.

```
template <typename T>
class Ptr
{
  public:
    Y* operator->() { return p; } // dereference to access member
    Y& operator*() { return *p; } // dereference to access whole object
    Y& operator[](int i) { return p[i]; } // dereference to access element
};
```
If you provide more than one of these operators, it might be wise to provide equivalence.

Overloading `->` is important to representing *indirection*. e.g. Iterators (Ch. 33)

There's no way of overloading operator `.` (dot).

## Increment and Decrement.

```
Ptr& operator++(); // prefix
Ptr operator++(int); // postfix
```

## Allocation and Deallocation, `operator new()` `operator delete()`

when `new` needs memory on free store for an object of type `X`, 

```
void operator new(sizeof(X));
```
When `new` needs memory on free store for an array of `N` objects of type `X`, 
```
void operator new[](N * sizeof(X));
```
Replacing global `operator new()` and `operator delete()` is not for fainthearted and not recommended.

Better approach is to supply these operations for a specific class. This class might be the base for many derived classes.

e.g. class that provides specialized allocator and deallocator for itself and all of its derived classes:
```
class Employee
{
  public:

    void* operator new(size_t);
    void operator delete(void*, size_t);

    void* operator new[](size_t);
    void operator delete[](void*, size_t);
};
```

Member `operator new()`, `operator delete()` are implicitly `static` members. 
Thus, they don't have a `this` pointer, and don't modify an object. They provide storage that a ctor can initialize and a destructor can clean up.

## User-defined Literals, `operator""`

e.g. 
```
constexpr complex<double> operator"" i(long double d) // imaginary literal
{
  return {0, d}; // complex is a literal type
}
```

compiler always checks for a suffix

4 kinds of literals that can be suffixed to make a user-defined literal (iso.2.14.8)

* integer literal (Sec. 6.2.4.1)
* floating-point literal (Sec. 6.2.5.1)
* string literal (Sec. 7.3.2)
* character literal (Sec. 6.2.3.2)

To get a C-style string from the program source text into a literal operator, we request both string and its number of characters. e.g.
```
string operator""s(const char* p, size_t n);

string s12 = "one two"s; // calls operator""("one two", 7)
string sxx = R"(two\ntwo)"s; // calls operator ""("two\\ntwo", 8)
```

A literal operator converting numerical values to strings could be quite confusing.

*template literal operator* is literal operator that takes its argument as a template parameter pack, rather than as a function argument. e.g.
```
template<char ...>
constexpr int operator""_b3(); // base 3, i.e. ternary
```

Variadic template techniques (Sec. 28.6) can be disconcerting, but it's the only way of assigning nonstandard meanings to digits at compile time.

## Example: A String Class.

cf. pp. 562 Sec. 19.3.1 Essential Operations. Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

`String` has value semantics, i.e. after an assignment `s1 == s2`, the 2 strings `s1` and `s2` are fully distinct, and subsequent changes to one have no effect on the other. 
Alternative would be to give `String` pointer semantics. That would let changes to `s2` after `s1 = s2` also affect the value of `s1`.

Where it makes sense, prefer value semantics, e.g. `complex`, `vector`, `Matrix` and `string`; however, for value semantics to be affordable, we need to pass `String`s by reference when we don't need copies and to implement move semantics (Sec. 3.3.2, Sec. 17.5.2) to optimize `return`s.

# Union declaration

cf. https://en.cppreference.com/w/cpp/language/union

A union is a special class type that can hold only one of its non-static data members at a time. 

A union can have member functions (including ctors, dtors), but not virtual functions.

A union can't have base classes, can't be used as a base.

A union can't have data members of reference types.

Since C++11,
If a union contains a non-static data member with non-trivial data member with non-trivial special member function (copy/move ctor, etc.), that function is deleted by default in the union and needs to be defined explicitly by programmer.
Since C++11,
If union contains non-static data member with non-trivial default ctor, default ctor of union deleted by default, unless variant member of union has a default member initializer.

The union is **only as big as necessary to hold its largest data member.**  
It's undefined behavior to read from member of union that wasn't most recently written.
```
union S
{
  std::int32_t n; // occupies 4 bytes
  std::uint16_t s[2]; // occupies 4 bytes
  std::uint8_t c; // occupies 1 byte
}; // the whole union occupies 4 bytes
```

# `std::strcpy` 


`<cstring>`
```
char* strcpy(char* dest, const char* src);
```

Copies character string pointed to by `src`, including the null terminator, to the character array whose first element is pointed to by `dest`.

Behavior is undefined if the `dest` array is not large enough.
Behavior is undefined if strings overlap.

# `std::memcpy`
`<cstring>`
```
void* memcpy(void* dest, const void* src, std::size_t count);
```
Copies count bytes from the object pointed to by `src` to the object pointed to by `dest`. Both objects are reinterpreted as arrays of `unsigned char`.

If the objects overlap, the behavior is undefined.
If either `dest` or `src` is a null pointer, the behavior is undefined, even it count is zero.
If objects are not "TriviallyCopyable", behavior of `memcpy` is not specified and may be undefined.

# `std::strlen`

`<cstring>`
```
std::size_t strlen(const char* str);
```
Returns length of the given byte string,
i.e. number of characters in a character array whose 1st element is pointed to by `str` up to and not including the first null character.

- behavior is undefined if there is no null character in the character array pointed to by `str`

cf. Sec. 43.5 has standard-library `memcpy()` to copy the bytes of the source into the target. That's a low-level and sometimes pretty nasty function. It should be used only where there are no objects with ctors or dtors in the copied memory because `memcpy()` knows nothing about types.

cf. pp. 568 Sec. 19.3.4 Member Functions. Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.


strong exception guarantee (Sec. 13.2)

# Friends; `friend` 

cf. pp. 571 Sec. 19.4 Friends. Ch. 19 *Special Operators*; Bjarne Stroustrup, **The C++ Programming Language**, 4th Ed.

An ordinary member function declaration specifies 3 logically distinct things:

1. function *can access the private part* of the class declaration.
2. function is in scope of the class
3. function must be invoked on an object (has a `this` pointer)

`static` *can only give 1st. 2 properties*.
`friend` give it *1st property only*.

So `friend` member function granted access to implementation of class, but is otherwise independent of that class.

pp. 571-572 example of `Matrix` on `Vector` multiplication (EY: 20181018); **implement this**

```
class Matrix;

class Vector {
  // ...
  friend Vector operator*(const Matrix&, const Vector&);
};

class Matrix
{
  friend Vector operator*(const Matrix&, const Vector&);
};
```
`friend` declaration can be placed in either private or public part of class declaration; doesn't matter where. 
  - Like a member function, a friend function is explicitly declared in declaration of the class of which it's a friend; it's therefore as much a part of that interface as is a member function.

- *member function of 1 class* can be the friend *of another*

```
class ListIterator { int* next(); };

class List
{
  friend int* ListIterator::next();
}
```
There's a shorthand for making *all functions of 1 class friends of another*:
```
class List
{
  friend class ListIterator;
};
```
This `friend` declaration makes all of `List_iterator`'s member functions friends of `List`. 

Declaring class a `friend` grants access to every function of that class. That implies we can't know the set of functions that can access granting class's representation just by looking at the class itself. 
  - in this, friend class declaration differs from declaration of a member function and friend function. 
  - Clearly, friend classes should be used with caution and only to express closely connected concepts.

Make a *template argument* a `friend`:

```
template <typename T>
class X
{
  friend T;
  friend class T; // redundant "class"
  // ...
};
```
