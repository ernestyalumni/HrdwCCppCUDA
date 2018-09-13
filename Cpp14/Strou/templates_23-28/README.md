# Templates, Generic Programming

## Templates

cf. pp. 665, Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

### A Simple String Template, `./String/`

Consider a string of characters. A string is a class that holds characters and provides operations such as subscripting, concatenation, and comparison that we usually associate with the notion of a "string."

We want to represent the notion of "string" with minimal dependence on a specific kind of character. The definition of a string relies on the fact that a character can be copied, and little else (Sec. 24.3). Thus, we can make a more general string type by taking the string of `char` from Sec. 19.3 and making the character type a parameter.

`template<typename C>` prefix specifies template is being declared and that a type argument `C` will be used in declaration. After its introduction, `C` used exactly like other type names. 
  `C` is a *type* name, need not be name of a *class* `template<class C>`. 

Mathematicians will recognize `template<typename C>` as a variant of "for all C" or "for all types C" or even "for all C, such that C is a type." 
  If you think along those lines, you'll note that C++ lacks a fully general mechanism for specifying the required properties of a template parameter `C`, i.e. we **can't** say "for all C, such that ..." where the "..." is a set of requirements for `C`. 
i.e. C++ doesn't offer a direct way to say what kind of type a template argument `C` is supposed to be (Sec. 24.3).

The standard library provides template class `basic_string` that's similar to templatized `String` (Sec. 19.3, 36.3). In standard library, `string` is a synonym for `basic_string<char>` (Sec. 36.3).

### Defining a Template

cf. pp. 669, Ch. 23.2.1 Defining a Template. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. Ch. 23.

A class generated from a class template is a perfectly ordinary class. Thus, use of a template doesn't imply any run-time mechanisms beyond what's used for an equivalent "handwritten" class. In fact, using template can lead to decrease of code generated because code for a member function of a class template is only generated if that member is used. (Sec. 26.2.1).

A template is a specification of how to generate something given suitable template arguments; the language mechanisms for doing that generation (instantiation (Sec. 26.2) and specialization (Sec. 25.3)) don't care much whether a class or function is generated. 

Members of a template class are themselves templates parametrized by parameters of their template class. When such a member is defined outside its class, it must explicitly be declared a template. e.g. 

```
template <typename C>
String<C>::String(): // String<C>'s constructor
  sz_{0}, ptr_{ch}
{
  ch[0] = {}; // terminating 0 of the appropriate character type
}

```

## Function Templates

cf. pp. 684, Ch. 23.5 Function Templates. Ch. 23. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.

When a function template is called, the *types of the function arguments* determine which version of the template is used; i.e. the template arguments are deduced from the function arguments. 


### Template Instantiation

cf. pp. 671, Ch. 23.2.2 Template Instantiation. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. Ch. 23.

*template instantiation* (Sec. 26.2) - process of generating a class or a function from a template plus a template argument list.

In general, it's the implementation's job - *not* the programmer's - to ensure that specializations of a template are generated for each template argument list used.


## Template argument deduction (TAD)

### Reference deduction by template argument deduction

cf. pp. 688, Ch. 23.5.2.1 Reference Deduction. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. Ch. 23.

It can be useful to have different actions taken for lvalues and rvalues. 

e.g. `XRef.h`, `XRef_main.cpp` 

``` 
template <typename T>
class XRef
{
  public:

    XRef(const int i, T* p): // store a pointer: Xref is the owner
      index_{i}, elem_{p}, owned_{true}
    {}

    XRef(int i, T& r):
      index_{i}, elem_{&r}, owned_{false}
    {}

    XRef(int i, T&& r):
      index_{i}, elem_{new T{std::move(r)}}, owned_{true}
    {}

    ~XRef()
    {
      if (owned_)
      {
        delete elem_;
      }
    }

    // Accessors
    int index() const
    {
      return index_;
    }

    const T elem() const
    {
      return *elem_;
    }

  private:
    int index_;
    T* elem_;
    bool owned_;

}; // END of class XRef
```

``` 
  std::string x {"There and back again"};

  XRef<std::string> r1 {7, "Here"};       // r1 owns a copy of string{"Here"}
  XRef<std::string> r2 {9, x};            // r2 just refers to x
  XRef<std::string> r3 {3, new std::string{"There"}};   // r3 owns the string{"There"}
```
`r1` picks `XRef(int, std::string&&)` because `"Here"` is a rvalue. 
`r2` picks `XRef(int, std::string&)` because `x` is a lvalue.

Lvalues and rvalues are distinguished by *template argument deduction*: an lvalue of type `X` is deduced as an `X&` and an rvalue as `X`. (or is it `T`, `T&`)

This differs from binding of values to non-template argument rvalue references (Sec. 12.2.1) but is especially useful for argument forwarding (Sec. 35.5.1). 

``` 
template <typename T>
  T&& std::forward(typename remove_reference_t<T>& t) noexcept; // Sec. 35.5.1

template <typename T>
  T&& std::forward(typename remove_reference_t<T>&& t) noexcept;


```





## Generic Programming

cf. pp. 699, Ch. 24.1 Introduction. Ch. 24**Generic Programming** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Templates offer:
* ability to pass types (as well as values and templates) as arguments without loss of information. 
  - This implies excellent opportunities for inlining, of which current implementations take great advantage. 
* delayed type checking (done at instantiation time). This implies opportunities to weave together information from different contexts. 
* ability to pass constant values as arguments. This implies ability to do compile-time computation.

*generic programming* - first and most common use of templates is to support *generic programming*, i.e. programming focused on design, implementation, and use of general algorithms, "general" meaning algorithm can be designed to accept a wide variety of types as long as they meet algorithm's requirements on its arguments.

templates provide (compile-time) parametric polymorphism. Type checking provided for templates checks use of arguments in the template definition, ratherh than against explicit interface (in a template delcaration) - we operate on values, and presence and meaning of an operation depend solely on its operand values.

template programming run-time cost is 0, and errors that, in a run-time typed language manifest themselves as exceptions, become compile-time errors in C++.

*Lifting* - generalizing an algorithm to allow greatest (reasonable) range of argument types 
*Concepts* - carefully specifying requirements of an algorithm (or class) on its arguments.

### Lifting; Algorithms and Lifting

cf. pp. 700, Ch. 24.2 Algorithms and Lifting. Ch. 24**Generic Programming** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

How do we get from a function doing specific operations on specific data to an algorithm doing more general operations on a variety of data types? 

The most effective way of getting a good algorithm is to generalize from one - and preferably more - concrete example; such a generalization is called *lifting*.

### SFINAE, Class template SFINAE, Function template SFINAE

[Class template SFINAE C++patterns](https://cpppatterns.com/patterns/class-template-sfinae.html) 

Conditionally instantiate a class template depending on template arguments.

``` 
#include <type_traits> 

template <typename T, typename Enable = void>
class foo; 

template <typename T> class foo<T, typename std::enable_if<std::is_integral<T>::value>::type> 
{ }; 

template <typename T> class foo<T, typename std::enable_if<std::is_floating_point<T>::value>::type> { };
```

If you want to simply prevent a template from being instantiated for certain template arguments, consider using `static_assert` instead.


[Function template SFINAE](https://cpppatterns.com/patterns/function-template-sfinae.html)

Conditionally instantiate a function template depending on template arguments.

``` 
#include <type_traits> 
#include <limits> 
#include <cmath> 

template <typename T> 
typename std::enable_if<std::is_integral<T>::value, bool>::type   
equal(T lhs, T rhs) 
{   
  return lhs == rhs; 
  } 

template <typename T> 
typename std::enable_if<std::is_floating_point<T>::value, bool>::type   
equal(T lhs, T rhs) 
{   
  return std::abs(lhs - rhs) < 0.0001; 
}
```


