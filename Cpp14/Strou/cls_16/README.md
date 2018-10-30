# Classes

cf. pp. 449, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

## Class Basics

cf. pp. 451, Sec. 16.2 Class Basics, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

* Members are accessed using `.` (dot) for objects and `->` (arrow) for pointers.
* Operators, such as `+`, `!`, and `[]` can be defined for a class.
* A class is a namespace containing its members.
* `public` members provide class's interface; `private` members provide implementation details.
* `struct` is a `class` where members are by default `public`.


cf. pp. 454, Sec. 16.2.4 `class` and `struct`, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

A class definition can be replicated in different source files using `#include` without violating the one-definition rule (Sec. 15.2.3).

I tend to use `struct` for classes that I think of as "just simple data structures."

If I think of a class as "a proper type with an invariant", I use `class`.

By default, members of a class are private.

I recommend the `{}` notation over `()` notation for **initialization** because it's explicit about what's being done (initialization), 
avoids some potential mistakes, and 
can be used consistently (Sec. 2.2.2, Sec. 6.3.5). 
There are cases where `()` notation must be used (Sec. 4.4.1, Sec. 17.3.2.1) but they're rare.

cf. pp. 457, Sec. 16.2.6 `explicit` Constructors, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

By default, a ctor invoked by a single argument acts as an implicit conversion from its argument type to its type.

Fortunately, we can specify that a ctor is not used as an *implicit* conversion. 
A ctor declared with keyword `explicit` can only be used for initialization and explicit conversions.


An initialization with an `=` is considered a *copy initialization*. In principle, a copy of initializer is placed into initialized object. However, such a copy may be optimized away (elided), and a move operation (Sec. 3.3.2, Sec. 17.5.2) may be used if initializer is a rvalue (Sec. 6.4.1).
Leaving out `=` makes initialization explicit. 

By default, declare a constructor that can be called with a single argument `explicit`. You need a good reason not to do so; if you define an implicit ctor, it's best to document your reason or a maintainer may suspect that you were forgetful (or ignorant).

## Physical and Logical Constness, `mutable`

cf. pp. 462, Sec. 16.2.9.2. Physical and Logical Constness, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Occasionally, a member function is logically `const`, but it still needs to change the value of a member. That is, to a user, function appears not to change state of its object, but
some detail that user cannot directly observe is updated.

This is called *logical constness*.

e.g. the `Date` class might have a function returning a string representation. Constructing this representation could be relatively expensive. Therefore, it'd make sense to keep a copy so that repeated requests would simply return the copy, unless `Date`'s value had been changed. 

```
class Date
{
  public:
    // ...
    string string_rep() const;   // string representation
  private:
    bool cache_valid;
    string cache;
    void compute_cache_value();   // fill cache
    // ...
};

```

From a user's point of view, `string_rep` doesn't change state of its `Date`, so it clearly should be a `const` member function. On the other hand, `cache` and `cache_valid` members must change occasionally for design to make sense. 

We can define a member of a class to be `mutable`, meaning, it can be modified even in a `const` object.

The programming techniques that support a cache generalize to various forms of lazy evaluation.

Note that `const` doesn't apply (transitively) to objects accessed through pointers or references. 
The human reader may consider such an object as "a kind of subobject," but compiler doesn't know such pointers or references to be any different from any others. 
That is, a member pointer doesn't have any special semantics that distinguish it from other pointers.

## Self-Reference

cf. pp. 465, Sec. 16.2.10. Physical and Logical Constness, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

For a set of related update functions, it's often useful to return a reference to the updated object so operations can be chained. For example, we'd like to write:

```
void f(Date& d)
{
  d.add_day(1).add_month(1).add_year(1);
}

```

To do this, each function must be declared to return a reference to a `Date`:

```
class Date
{
  public:

    Date& add_year(int n)
    {
      this->y_ +=n;
      return *this;
    }
    Date& add_month(int n);
    Date& add_day(int n);
};
```
Each (non-`static`) member function knows for which object it was invoked and can explicitly refer to it.

`*this` refers to object for which a member function is invoked.

In a non-`static` member function, `this` is a pointer to the objec for which the function was inovked.
  - in a non-`const` member function of class `X`, type of `this` is `X*`
  - However `this` is considered an rvalue, so it's not possible to take the address of `this` or to assign to `this`
  - in a `const` member function of class `X`, type of `this` is `const X*`, to prevent modification of the object itself (see also Sec. 7.5)

Most uses of `this` are implicit; in particular, every reference to a non-`static` member from within a class relies on an implicit use of `this` to get the member of the appropriate object, e.g. `this->y_` (but tedious).

One common explicit use of `this` is in linked-list manipulation. (???)

cf. pp. 466, Sec. 16.2.10. Physical and Logical Constness, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 


## Member access

cf. pp. 467, Sec. 16.2.11. Member Access, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

```
int (S::*) pmf() {&S::f}; // X's member f
```
That last construct (a pointer to member) is fairly rare and esoteric; see Sec. 20.6

## `static`

cf. pp. 467, Sec. 16.2.12. [static] Members, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

`static` member - a variable that's part of a class, yet is not part of an object of that class.
  - There is exactly 1 copy of a `static` member instead of 1 copy per object, as for ordinary non-`static` members (Sec. 6.4.2).


In multi-threaded code, `static` data members require some kind of locking or access discipline to avoid race conditions (Sec. 5.3.4, Sec. 41.2.4). 
Since multi-threading is now very common, it's unfortunate that use of `static` data members was quite popular in older code.

## Member Types 

cf. pp. 469, Sec. 16.2.13. Member Types, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Types and type aliases can be members of a class.

A *member class* (often called a *nested class*) can refer to types and `static` members of its enclosing class.
- It can only refer to non-`static` members when it's given an object of the enclosing class to refer to.

A nested class has access to members of its enclosing class, even to `private` members (just as a member function has), but has no notion of a current object of the enclosing class.

A class does not have any special access rights to the members of its nested class. 

Member classes are more a notational convenience than a feature of fundamental important.
On the other hand, member aliases are important as the basis of generic programming techniques relying on associated types (Sec. 28.2.4, Sec. 33.1.3). 
Member `enum`s are often an alternative to `enum class`es when it comes to avoiding polluting an enclosing scope with names of enumerators (Sec. 8.4.1).

### nested classes

cf. https://en.cppreference.com/w/cpp/language/nested_types

**nested classes** - declaration of a class/struct or union may appear within another class.

Name of the nested class exists in scope of enclosing class, and name lookup from a member function of a nested class visits scope of the enclosing class after examining the scope of the nested class.
Like any member of its enclosing class, nested class has access to all names (private, protected, etc.) to which enclosing class has access, but is otherwise independent and has no special access to the `this` pointer of the enclosing class.

Declarations in a nested class can use any members of the enclosing class, following usual usage rules for non-static members.

`friend` functions defined within a nested class have no special access to members of the enclosing class even if lookup from body of a member function that's defined within a nested class can find private members of the enclosing class.

Out-of-class definitions of members of a nested class appear in namespace of enclosing class.

Nested classes can be forward-declared and later defined, either within enclosing class body, or outside of it.

See `nested_cls_main.cpp`.

## Concrete Classes

cf. pp. 470, Sec. 16.3. Concrete Classes, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

A class is called *concrete* (or a *concrete class*) if its representation is part of its definition.

This distinguishes it from abstract classes (Sec. 3.2.2, Sec. 20.4) which provide an interface to a variety of implementations.

Having the representation available allows us:
* To place objects on the stack, in statically allocated memory, and in other objects
* To copy and move objects (Sec. 3.3, Sec. 17.5)
* To refer directly to named objects (as opposed to accessing through pointer and references)

cf. pp. 470-471, Sec. 16.3 Concrete Classes, Ch. 16 **Classes** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Set of operations is fairly typical for a user-defined type:

1. A ctor specifying how objects/variables of the type are to be initialized (Sec. 16.2.5).
2. Set of functions allowing a user to examine a Date. These functions are marked `const` to indicate that they don't modify the state of the object/variable for which they are called.
3. Set of functions allowing user to modify `Date` without actually having to know the details of the representation or fiddle with intricacies of the semantics.
4. Implicitly defined operations that allow `Date`s to be freely copied (Sec. 16.2.2)
5. A class, `Bad_date`, used for reporting errors as exceptions.
6. Set of useful helper functions. Helper functions aren't members and have no direct access to the representation of a `Date`, but they're identified as related by use of the namespace `Chrono`.

