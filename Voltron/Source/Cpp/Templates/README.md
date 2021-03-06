cf. David Vandevoorde, Nicolai M. Josuttis, Douglas Gregor. **C++ Templates: The Complete Guide.** (2nd Edition). *Addison-Wesley Professional*; 2 edition (September 18, 2017). ISBN-13: 978-0321714121

# Function Templates

cf. Ch. 1, VJG. pp. 3

Type parameter `T` - values of type `T` since C++17, can pass temporaries (rvalues, see Appendix B) even if neither copy nor move ctor is valid.

Templates aren't compiled into single entitites that can handle any type. Instead, different entities are generated from template for every type for which the template is used.

Process of replacing template parameters by concrete types is called **instantiation**. It results in an **instance** of a template.
- Note that mere use of function template can trigger such an instantiation process.

Note also that `void` is a valid template argument provided resulting code is valid.

## 2-Phase Translation

cf. 1.1.3 Two-Phase Translation, VJG (2017), pp. 6

An attempt to instantiate a template for a type that doesn't support all operations used within it will result in compile-time error.

Templates are "compiled" in 2 phases:
1. Without instantiation at *definition time*, template code itself is checked for correctness ignoring template parameters. Includes:
  - Syntax errors, 
  - Using unknown names (type names, function names, ...) that don't depend on template parameters are discovered.
  - Static assertions that don't depend on template parameters are checked.
2. At *instantiation time*, template code checked (again) to ensure all code is valid. That is, now especially, all parts that depend on template parameters are double-checked.

e.g.

```
template <typename T>
void foo(T t)
{
  undeclared(); // first-phase compile-time error if undeclared() unknown
  undeclared(t); // second-phase compile-time error if undeclared(T) unknown
  static_assert(sizeof(int) > 10, // always fails if sizeof(int) <= 10
    "int too small");
  static_assert(sizeof(T) > 10, // fails if instantiated for T with size <= 10
    "T too small");
}
```
Fact that names are checked twice is called *two-phase lookup* (Sec. 14.3.1 on pp. 249)

Note, some compilers don't perform full checks of 1st. phase. So might not see general problems until template code is instantiated at least once.

### Compiling and Linking

2-phase translation leads to important problem: When a function template is used in a way that **triggers its instantiation**, a compiler will (at some point) need to *see that template's definition.*
* This breaks usual compile and link distinction for ordinary functions, when declaration of function is *sufficient to compile its use*. Ch. 9

## Template Argument Deduction

cf. 1.2 Template Argument Deduction, VJG (2017), pp. 7

When we call a function template for some arguments, template parameters are determined by the arguments we pass.

However, `T` might only be "part" of the type. e.g. if we declare `max()~` to use constant references:
```
template <typename T>
T max (T const& a, T const& b)
{
  return b < a ? a : b;
}
```
`T` again deduced as `int`, because function parameters match for `int const&`.

### Type Conversions During Type Deduction

Note that automatic type conversions are limited during type deduction:
* When declaring call parameters by reference, even trivial conversions don't apply to type deduction.
  - 2 arguments declared with same template parameter `T` must match *exactly*.
* When declaring call parameters by value, only trivial conversion that *decay* are supported: 
  - Qualifications with `const` or `volatile` are ignored, references convert to the referenced types,
  - raw arrays or functions convert to corresponding pointer type
  - For 2 arguments declared with same template parameter `T` the *decayed* types must match

However, following are errors:


  // ERROR: T can be deduced as int or double.
  //max(4, 7.2);

  std::string s;

  // ERROR: T can be deduced as char const[6] or std::string
  //max("hello", s);

  // 3 ways to handle such errors:

  // 1. Cast the arguments so that they both match:
  BOOST_TEST(max(static_cast<double>(4), 7.2) == 7.2); // OK

  // 2. Specify (or qualify) explicitly the type of T to prevent from attempting
  // type deduction:
  BOOST_TEST(max<double>(4, 7.2) == 7.2);

  // 3. Specify that parameters may have different types.
  
  BOOST_TEST(true);

Sec. 1.3, pp. 9, will elaborate on these options. Sec. 7.2 on pp. 108 and Ch. 15 will discuss rules for type conversions during type deduction in detail.


#### Type Deduction for Default Arguments

Type deduction doesn't work for default call arguments.

To support this case, you also have to declare a default argument for template parameter, which will be discussed in Sec. 1.4 on pp. 13.

### Multiple Template Parameters

So far, function templates have 2 distinct sets of parameters:
1. *Template parameters*, which are declared in angle brackets before function template name
2. *Call parameters*, which are declared in parentheses after function template name:
```
T max(T a, T b) // a and b are call parameters
```

If you use 1 of the parameter types as return type, argument for other parameter might get converted to this type, regardless of caller's intention.

different ways to deal with this problem:
* Introduce 3rd template parameter for return type.
* Let compiler find out return type
* Declare return type to be "common type" of 2 parameter types.

#### Template Parameters for Return Types
pp. 10. Ch. 1: Function Templates, VJG

Template argument deduction doesn't take return types into account
- Deduction can be seen as part of overload resolution - a process that's not based on selction of return types. 
  * sole exception is return type of conversion operator members.

If template argument, e.g. `RT` doesn't appear in types of function call parameters, in C++, return type can't be deduced from context in which caller uses the call.

cf. pp. 11, Sec. 1.3.2 Deducing the Return Type. If return type depends on template parameters, simplest and best approach to deduce return type is to let compiler find out.

cf. pp. 12, VJG (2017).

Note that initialization of type `auto` **always decays**. This also applies to return values when return type is just `auto`. `auto` as return type behaves just as in following code, where `a` is declared by decayed type of `i, int`. 


### Overloading Function Templates

A nontemplate function can coexist with a function template that has same name and can be instantiated with same type. All other factors being equal, overload resolution process prefers nontemplate over one generated from template.

# Nontype (Non-Type) template parameters

```
// Pointers to functions.
// \ref https://en.cppreference.com/w/cpp/language/template_parameters
template<typename X, typename FX, FX (*ObjectMap)(const X&)>
FX object_map(const X& x)
{
  return (*ObjectMap)(x);
}
```

