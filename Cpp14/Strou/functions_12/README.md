# Functions

cf. pp. 305, Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

12.1 Function Declarations 

##  `inline`, `constexpr`, `noexcept`, `[[noreturn]]`, `virtual`, `override`, `final`, `static`, `const`; Parts of a Function Declaration  

cf. pp. 307 12.1.2 Parts of a Function Declaration Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

* `inline` - indicating desire to have function calls implemented by inline function body (Sec. 12.1.5)  
* `constexpr` - it should be possible to evaluate function at compile time if given constant expressions as arguments (Sec. 12.1.6)
* `noexcept` - function may not throw an exception (Sec. 13.5.1.1) 
* `static`, 1 example of a *linkage specification* (???)  
* `[[noreturn]]` - function won't return using normal call/return mechanism (12.1.4) 

In addition, member function may be specified as 
* `virtual`, can be overriden in a derived class (Sec. 20.3.2)  
* `override`, overriding a virtual function from a base class (Sec. 20.3.4.1) 
* `final`, indicating that it can't be overriden in a derived class (Sec. 20.3.4.2)  
* `static`, it's not associated with a particular object (Sec. 16.2.12)  
* `const`, indicating that it may not modify its object (Sec. 3.2.1.1, Sec. 16.2.9.1)  

If you feel inclined to give readers a headache, you may write something like 
```  
struct S {
    [[noreturn]] virtual inline auto f(const unsigned long int *const) -> void const noexcept;
};
```

Unfortunately, to preserve C compatibility, `const` ignored at the highest level of an argument type. For example, this is 2 declarations of same function:  
```  
void f(int);            // type is void(int) 
void f(const int);      // type is void(int)  
```  
That function, `f()`, could be defined as  
```  
void f(int x) { /* we can modify x here */ }
```
Alternatively, define `f()` as 
``` 
void f(const int x) { /* we can't modify x here */ }
```
Either case, argument that `f()` can or can't modify is a copy of what caller provided, so there's no danger of obscure modification of calling context.  

Function argument names aren't part of the function type and need not be identical in different declarations.  For example:
``` 
int& max(int& a, int& b, int& c); // return a reference to the larger of a, b, and c  

int& max(int& x1, int& x2, int& x3)
{
    return (x1 > x2) ? ((x1 > x3) ? x1 : x3) : ( (x2 > x3) ? x2 : x3);
}
```  
Naming arguments in declarations that aren't definitions is optional and commonly used to simplify documentation.  Conversely, we can indicate an argument is unused in a function definition by not naming it.  
```  
void search(table* t, const char* key, const char*)
{
    // no use of the third argument
}
```  

In addition to functions, that are few other things we can call; these follow most rules defined for functions, such as rules for argument passing (Sec. 12.2)  
* *Constructors* (Sec. 2.3.2, 16.2.5) aren't technically functions; particularly, they don't return a value, can initialize bases and members (Sec. 17.4), and can't have their address taken.
* *Destructors* (Sec. 3.2.1.2, Sec. 17.2) can't be overloaded and can't have their address taken. 
* *Function objects* (Sec. 3.4.3, Sec. 19.2.2) aren't functions (they're objects) and can't be overloaded, but their `operator()`s are functions.
* *Lambda expressions* (Sec. 3.4.3, Sec. 11.4) are basically a shorthand for defining function objects.  

## Returning Values

cf. pp. 307 12.1.4 Returning Values Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Every function declaration contains a specification of the function's *return type* (except for constructors and type conversion functions).  
Traditionally, in C, C++, return types comes first in a function declaration (before function name). However, function declaration can also be written, placing return type after argument list, e.g. following 2 declarations are equivalent:  
```  
string to_string(int a);            // prefix return type
auto to_string(int a) -> string;    // suffix return type
```  

Essential use for a suffix return types comes in **function template declarations** in which return type depends on arguments, e.g.  
```  
template<class T, class U>
auto product(const vector<T>& x, const vector<U>& y) -> decltype(x*y);
```  

Like semantics of argument passing, the semantics of function value return are identical to semantics of copy initialization (Sec. 16.2.6). A `return`-statement initializes a variable of the returned type.  
Type of a return expression is checked against type of returned type, and all standard and user-defined type conversions are performed. For example: 
``` 
double f() { return 1; }    // 1 is implicitly converted to double{1}
``` 

Each time a function is called, a new copy of its arguments and local (automatic) variables is created. The store is reused after function returns, so pointer to a local non-static variable shouldn't be ever returned.  Contents of location pointed to will change unpredictably: 

e.g.  

```  
int* fp()
{
    int local = 1;
    // ...
    return &local; //
}
```  

There are no `void` values.  However, call of a `void` function may be used as return value of a `void` function. e.g. 
```  
void g(int* p);

void h(int* p)
{
    // ...
    return g(p);    // OK: equivalent to "g(p); return;"
}
```  
This form of return is useful to avoid special cases when writing template functions where return type is a template parameter. 

## `inline` functions 

cf. 12.1.5 `inline` Functions

A function can be defined to be `inline`. For example:
```
inline int fac(int n)
{
    return (n < 2) ? 1 : n * fac(n-1);
}
```
`inline` specifier is a hint to compiler that it should attempt to generate code for a call of `fac()` inline rather than laying down the code for the function once and then calling through the usual function call mechanism.  

A clever compiler can generate constant `720` for a call `fac(6)`.  But, the possibility of mutually recursive inline functions, inline functions that recurse or not depend on input, etc., makes it impossible to guarantee every call of an inline function is actually inlined.  
So 1 compiler might generate `720`, another `6*fac(5)`, yet another an un-inlined call `fac(6)`.  
If you want to guarantee a value is computed at compile time, declare it `constexpr` and make sure that all functions used in its evaluation are `constexpr`.  

To make inlining possible in absence of unusually clever compilation and linking facilities, the definition, and not just declaration, of an inline function must be in scope (Sec. 15.2).  An inline specifier doesn't affect the semantics of a function; in particular, an inline function still has an unique address, and so do `static` variables (Sec. 12.1.8) of an inline function.  

If an inline function is defined in more than 1 translation unit (e.g. typically because it was defined in a header, Sec. 15.2.2), its definition in the different translation units must be identical (Sec. 15.2.3). 

## `constexpr` 

cf. pp. 312 12.1.6 `constexpr` Functions Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

In general, a function can't be evaluated at compile time and therefore can't be called in a constant expression (Sec. 2.2.3, Sec. 10.4).   
By specifying a function `constexpr`, we indicate that we want it to be usable in constant expressions if given constant expressions as arguments.  
```  
constexpr int fac(int n)
{
    return (n > 1) ? n * fac(n-1) : 1;
}

``` 

To be evaluated at compile time, a function must be suitably simple: a `constexpr` function must consist of 
* a single `return`-statement; no loops and no local variables allowed. 
* `constexpr` function may not have side effects (i.e. a `constexpr` function is a pure function); e.g.  

```  
int glob;

constexpr void bad1(int a)  // error: constexpr function can't be void
{
    glob = a;               // error: side effect in constexpr function
}

constexpr int bad2(int a)
{
    if (a >= 0)
    {
        return a;
    }
    else
    {
        return -a;  // error: if-statement in constexpr function
    }
}

constexpr int bad3(int a)
{
    sum = 0;                   // error: local variable in constexpr function 
    for (int i = 0; i < a; i+=1)
    {
        sum += fac(i);          // error: loop in constexpr function
    }
    return sum;
}

```  
EY : 20180304 But I tried this in [`./constexpr.cpp`]() and it works with C++17 (`-std=c++17`) and g++ 5.  WTH?


`constexpr` function allows recursion and conditional expressions.  However, debugging gets unnecessarily difficult and compile times longer unless you restrict use of `constexpr` functions to relatively simple tasks for which they're intended.  

By using literal types (Sec. 10.4.3), `constexpr` functions can be defined to use user-defined types.  

Like inline functions, `constexpr` functions obey ODR ("one-definition rule"), so definitions in different translation units must be identical (Sec. 15.2.3).  
You can think of `constexpr` functions as a restricted form of inline functions (Sec. 12.1.5).  

### `constexpr` and References  

cf. pp. 312 12.1.6.1 `constexpr` and References Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

`constexpr` function can't have side effects, so writing to nonlocal objects isn't possible.  
However, `constexpr` function can refer to nonlocal objects as long as it doesn't write to them.  

`constexpr` function can take reference arguments. Of course, it can't write through such references, but `const` reference parameters are as useful as ever.  e.g., in std lib (Sec. 40.4), 

```  
template<> class complex<float> {
    public:
    // ...
        explicit constexpr complex(const complex<double>&);
        // ...
};
```
This allows us to write: 
``` 
constexpr complex<float> z {2.0};
```
The temporary variable that's logically constructed to hold the `const` reference argument simply becomes a value internal to the compiler.  

It's possible for a `constexpr` function to return a reference or a pointer.  e.g.  
```  
constexpr const int* addr(const int& r) { return& r; } // OK
```  

However, doing so brings us away from the fundamental role of `constexpr` functions as part of constant expression evaluation.  

