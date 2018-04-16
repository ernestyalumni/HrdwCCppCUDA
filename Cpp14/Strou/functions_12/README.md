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
Particularly, it can be tricky to determine whether the result of such a function is a constant expression. Consider:  
```  
static const int x = 5;
constexpr const int* p1 = addr(x);      // OK
constexpr int xx = *p1;                 // OK

static int y; 
constexpr const int* p2 = addr(y);      // OK
constexpr int yy = *y;                  // error: attempt to read a variable 

constexpr const int* tp = addr(5);      // error: address of temporary
```  

### Conditional Evaluation (in `constexpr`)

cf. pp. 313 12.1.6.2 Conditional Evaluation Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

### `[[noreturn]]` Functions  

`[[...]]` is called an *attribute* and can be placed just about anywhere in C++ syntax.  
In general, an attribute specifies some implementation-dependent property about the syntactic entity that precedes it.  
There are only 2 standard attributes (Sec. iso.7.6) 
* `[[noreturn]]` 
* `[[carries_dependency]]` 

Placing `[[noreturn]]` at start of a function declaration indicates function isn't expected to return.  e.g. 
```  
[[noreturn]] void exit(int);
```  

### Local Variables  

cf. pp. 314 12.1.8 Local Variables Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

A name defined in a function is commonly referred to as a *local name*.  
A local variable or constant is initialized when a thread of execution reaches its definition.  
Unless declared `static`, each invocation of the function has its own copy of the variable.  
If local variable declared `static`, a single, statically allocated object (Sec. 6.4.2) will be used to represent that variable in all calls of the function.  e.g.
```
void f(int a)
{
    while (a--) {
        static int n = 0;   // initialized once
        int x = 0;          // initialized 'a' times in each call of f()  

        cout << "n == " << n++ << ", x == " << x++ << '\n';
    }
}

int main()
{
    f(3);
}
```  

This prints:
```  
n == 0, x == 0 
n == 1, x == 0
n == 2, x == 0 
```  

Initialization of `static` local variables doesn't lead to a data race (Sec. 5.3.1) unless you enter function containing it recursively or a deadlock occurs (Sec.iso.6.7).  That is, the C++ implementation must guard the initialization of a local `static` variable with some kind of lock-free construct (e.g., a `call_once`, Sec. 42.3.3).  
Effect of initializing a local `static` recursively is undefined.  e.g.  

```  
int fn(int n)
{
    static int n1 = n;      // OK
    static int n2 = fn(n-1) + 1; // undefined
    return n;
}
```

`static` local variable allows function to preserve information between calls without introducing a global variable that might be accessed and corrupted by other functions (cf. Sec. 16.2.12)  

`static` local variable is useful for avoiding order dependencies among nonlocal variables (sec. 15.4.1)

There are no local functions; if you need 1, use a function object or a lambda expression (Sec. 3.4.3, 11.4)

### Argument Passing  

cf. pp. 316 12.2 Argument Passing Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

When a function is called (using suffix `()`, known as the *call operator* or *application operator*), store is set aside for its *formal arguments* (also known as its *parameters*); each formal argument initialized by its corresponding actual argument.   
Semantics of argument passing are identical to semantics of initialization (copy initialization, to be precise, Sec. 16.2.6).  
Type of an actual argument is checked against type of the corresponding formal argument, and all standard and user-defined type conversions are performed.  
Unless a formal argument (parameter) is a reference, a copy of the actual argument is passed to the function.  

#### `const` reference (argument passing)  

argument might be declared `const` reference to indicate that the reference is used for efficiency reasons only, and not to enable called function to change value of the object:

Absence of `const` in declaration of reference argument taken as statement of intent to modify the variable.  

Similarly, declaring pointer argument `const` means that value of an object pointed to by that argument isn't changed by the function. 

Following rules for reference initialization, a literal, a constant, and an argument that requires conversion can be passed as a `const T&` argument, but not as a plain (non-`const`) `T&` argument.  
Allowing conversions for a `const T&` argument ensures that such an argument can be given exactly the same set of values as a `T` argument by passing the value in a temporary, if necessary.  

#### 12.2.2 Array Arguments
cf. pp. 318 12.2.2 Array Arguments Passing Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Arrays differ from other types in that an array isn't passed by value.  Instead, a pointer is passed (by value).  

A parameter of array type is equivalent to a parameter of pointer type.  
```  
void odd(int* p);
void odd(int a[]);
void odd(int buf[1020]);
```   
These 3 declarations are equivalent and declare the same function.  

Size of an array isn't available to called function. This is a major source of errors, but 
- a 2nd. argument specifying size can be passed, e.g. 
```
void compute1(int* vec_ptr, int vec_size); // one way
```
- preferable to pass a reference to some container, e.g. `vector`, `array`, `map`  

If you really want to pass an array, rather than a container, or pointer to 1st element of an array,  
declare a parameter of type reference to array.  

```
void f(int(&r)[4]);  

void g()
{
  int a1[] = {1,2,3,4};
  int a2[] = {1,2}; 

  f(a1);    // OK
  f(a2);    // error: wrong number of elements
}
```
Note number of elements is part of a reference-to-array type. That makes such references far less flexible than pointers and containers (such as `vector`). 

Main use of reference to arrays is in templates, where number of elements is then deduced. e.g. 
``` 
template <class T, int N> void f(T(&r)[N])
{
    // ...
}

int a1[10]; 
double a2[100];

void g()
{
  f(a1);        // T is int; N is 10
  f(a2);        // T is double; N is 100
}
```



#### Rules of thumb for passing arguments

1. Use pass-by-value for small objects
2. Use pass-by-`const`-reference to pass large values that you don't need to modify 
3. Return a result as `return` value rather than modifying an object through an argument 
4. Use rvalue references to implement move (Sec. 3.3.2, 17.5.2) and forwarding (Sec. 23.5.2.1).  
5. Pass a pointer if "no object" is a valid alternative (and represent "no object" by `nullptr`). 
6. Use pass-by-reference only if you have to.  




### Pointer to Function

cf. pp. 332 12.5 Pointer to Function Ch. 12 **Functions** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  
