# Derived Classes

cf. pp. 613, Ch. 21 **Class Hierarchies** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.

## Implementation Inheritance

cf. pp. 614, Sec. 21.2.1. Implementation Inheritance. Ch. 21 **Class Hierarchies** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.

Class hierarchy using implementation inheritance (commonly found in older programs).

Class `IValBox` defined basic interface to all `IValBox`es, and specified default implementation that more specific kinds of `IValBox` can override with their own versions.

```
# "Integer Value Input Box"
class IValBox
{
  protected:

    int val;
    int low, high;
    bool changed {false}; // changed by user using set_value()

  public:

    IValBox(int ll, int hh) : val {ll}, low {ll}, high {hh}
    {}

    virtual int get_value()
    {
      changed = false;
      return val;
    } // for application

    virtual void set_value(int i)
    {
      changed = true;
      val = i;
    } // for user

    virtual void reset_value(int i)
    {
      changed = false;
      val = i;
    } // for application

    virtual void prompt()
    {}

    virtual bool was_changed() const
    {
      return changed;
    }

    virtual ~IValBox() {};
}
```

### Critique of Implementation Inheritance

cf. pp. 616, Sec. 21.2.1.1 Critique.

* retrofitting base classes with another base class e.g. retrofitted `BBwidget` as base of `Ival_box`.
  - use of `BBwidget` isn't part of our basic notion of `Ival_box`
* every derived class shares basic data declared in base class (e.g. `Ival_box`).
  - That data is an implementation detail that crept into our `Ival_box` interface.
    * from a practical point of view, it's the wrong data; e.g. `Ival_slider` doesn't need value stored specifically; it can easily be calculated from position of the slider when someone executes `get_value()`.
* in general, keeping 2 related, but different, sets of data is asking for trouble
* in almost all cases, a protected interface should contain only functions, types, and constants
* deriving from `BBwidget` (i.e. deriving from another base class) means changes to class `BBwidget` may force users to recompile or even rewrite their code to recover from such changes
* having our user-interface systems "wired in" as the 1 and only base of our 1 and only `Ival_box` interface just isn't flexible

## Interface Inheritance

cf. pp. 617 Sec. 21.2.2 Interface Inheritance

hierarchy that solves problems presented in critique of implementation inheritance
1. user-interface system should be an implementation detail that's hidden from users who don't want to know about it
2. `Ival_box` class (base class) should contain no data
3. No recompilation of code using `Ival_box` family of classes should be required after change of user-interface system.
4. `Ival_box` for different interface systems should be able to coexist in our program.

1st, specify class `Ival_box` (base class) as pure interface ()

```
class Ival_box
{
  public:

    virtual int get_value() = 0;
    virtual void set_value(int i) = 0;
    virutal void reset_value(int i) = 0;
    virtual void prompt() = 0;
    virtual bool was_changed() const = 0;
    virtual ~lval_box() {}
};
```

*Data is gone, ctor gone since there's no data to initialize.*
*Added virtual destructor to ensure proper cleanup of data that'll be defined in derived class*

e.g.

```
class Ival_slider : public Ival_box, protected BBwidget
{
  public:

    Ival_slider(int, int);
    ~Ival_slider() override;

    int get_value() override;
    void set_value(int i) override;
    // ...

  protected:

    // ... funtions overriding BBwidget virtual functions
    // e.g., BBwidget::draw(), BBwidget::mouse1hit() ...

  private:

    // .. data needed for slider ...
};

```

Derived class inherits from abstract class that requires it to implement base class's pure virtual functions.
* can also inherit from (e.g. `protected BBwidget`), providing it means to implement
* since `Ival_box` provides interface for derived class, it's derived using `public`,
* since `BBwidget` only implementation aid, it's derived using `protected` (Sec. 20.5.2).
* Stroustrup used `protected` derivation instead of more restrictive (and usually safer) `private` derivation to make `BBwidget` available to classes derived from `Ival_slider`
* Stroustrup used explicit `override` because this "widget hierarchy" is exactly the kind of large, complicated hierarchy where being explicit can help minimize confusion.
* *multiple inheritance* (Sec. 21.3) - note `Ival_slider` must override functions from both `Ival_box` and `BBwidget`; therefore, it must be derived directly or indirectly from both

* note, making "implementation class" `BBWidget` a member of `Ival_box` is not a solution because a class can't override virtual functions of its members.
* representing window by a `BBwidget*` member in `Ival_box` leads to completely different design with a separate set of tradeoffs
  - -> hence, *multiple inheritance*

- "multiple inheritance" isn't complicated and scary; use of 1 base class for implementation details, another for interface (abstract class) is common to all languages suppporting inheritance and compile-time checked interfaces

- **Cleanup**:
  * many classes requires some form of cleanup for an object before it goes away.
  * since abstract class `Ival_box` can't know if a derived class requires such cleanup, it must assume that it does require some.
  * ensure proper cleanup by
    - **defining virtual destructor** `Ival_box::~Ival_box()` **in base** and overriding it suitably in derived classes.

e.g.
```
void f(Ival_box* p)
{
  // ...
  delete p;
}
```
`delete` operator explicitly destroys object pointed to by `p`.
- We have no way of knowing exactly which class object pointed to by `p` belongs, but thanks to `Ival_box`'s virtual dtor, proper cleanup as (optionally) defined by that class' destructor will be done

### Further Interface Inheritance; "Alternative Implementations" on top of Interface Inheritance

Insulate application-oriented `Ival_box` class from implementation details, derive abstract `Ival_slider` class from `Ival_box`, then derive system_specific `Ival_sliders` from that

```
class Ival_box
{ /* ... */ };

class Ival_slider : public Ival_box
{ /* ... */ };

class BB_ival_slider : public Ival_slider, protected BBwidget
{ /* ... */ };

class CW_ival_slider : public Ival_slider, protected CWwidget
{ /* ... */ };
```

*Usually, we can do better yet by utilizing more specific classes in implementation hierarchy*

EY : Make derived class for Implementations with multiple inheritance from abstract interface class, abstract implementation base class?

```
class BB_ival_slider : public Ival_slider, protected BBslider
{ /*... */ };

class CW_ival_slider : public Ival_slider, protected CWslider
{ /*... */ };
```

This improvement becomes significant where - as is not uncommon - our abstractions aren't too different from ones provided by system used for implementation.
  - in that case, programming reduced to mapping between similar concepts

### Critique of Interface Inheritance

cf. Sec. 21.2.3.1 Critique, pp. 622, Ch. 21 Stroustrup

abstract class design "good" : users of `Ival_box` abstract class application hierarchy can't accidentally use facilities from implementation because only facilities explicitly specified in `Ival_box` hierarchy are accessible; nothing is implicitly inherited from implementation-specific base class.

Logical conclusion is system **represented to users as hierarchy of abstract classes, and implemented by classical hierarchy**, i.e.
  * use abstract classes tosupprt interface inheritance (Sec. 3.2.3, Sec.20.1)
  * use base classeswith implementation of virtual functions to support implementtion inheritance (Sec. 3.2.3, Sec. 20.1)


