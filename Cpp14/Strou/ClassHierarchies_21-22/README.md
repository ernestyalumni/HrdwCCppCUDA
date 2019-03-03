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
*Added **virtual destructor to ensure proper cleanup of data** that'll be defined in derived class*

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
  * since abstract class `Ival_box` **can't know if a derived class requires such cleanup, it must assume that it does require some**.
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

### Localizing Object Creation (for Interface Inheritance)

Most of an application can be written using the `Ival_box` interface.
  - Further, shoud derived interfaces evolve to provide more facilities than plain `Ival_box`, then most of an application can be written using the `Ival_box`, `Ival_slider`, etc., interfaces.
Problem: creation of objects must be done using implementation-specific names such as `CW_ival_dial`, and `BB_flashing_Ival_slider`

We'd like to minimize number of places where such specific names occur, and object creation is hard to localize unless it's done systematically.

Solution: introduce, as usual, an indirection.

Simple way - introduce abstract class to represent set of creation operations:

```
class Ival_maker
{
  public:

    virtual Ival_dial* dial(int, int) = 0; // make dial
    virtual Popup_Ival_slider* popup_slider(int, int) = 0; // make popup slider
    // ...
};
```

For each interface from `Ival_box` family of classes that user should know about, class `Ival_maker` provides function that makes an object.

**factory** - e.g. `class Ival_maker`.
  - its functions are sometimes called *virtual constructors* (Sec. 20.3.6)

Now represent each user-interface system by class **derived** from `Ival_maker`: e.g.

```
class BB_maker : public Ival_maker // make BB versions
{
  public:

    Ival_dial* dial(int, int) override;
    Popup_Ival_slider* popup_slider(int, int) override;
    // ...
};
...

```

## Multiple Inheritance (redux, revisit)

cf. Sec. 21.3 *Multiple Inheritance*, Stroustrup

* *Shared interfaces* - leads to less replication of code using classes and making such code more uniform; often called *run-time polymorphism* or *interface inheritance*
* *Shared implementation* - leads to less code and more uniform implementation code; *implementation inheritance*

A class can **combine aspects of these 2 styles**.

### Multiple Interfaces

pp. 624 Sec. 21.3.1 *Multiple Interface*, Stroustrup.

Any class without mutable state can be used as an interface in a multiple-inheritance lattice without significant complications and overhead.

Key observation is that a class without mutable state can be replicated if necessary or shared if that's desired.

Use of multiple abstract classes as interfaces is almost universal in object-oriented designs (in any language with a notion of an interface).

### Multiple Implementation Classes

Consider simulation of bodies orbiting Earth, in which orbiting objects represented as object of class `Satellite`; `Satellite` object would contain orbital, size, shape, density parameters, etc., and provide operations for orbital calculations, etc. Dervied classes would add data members, functions, would override some of `Satellite`'s virtual functions.

Assume, graphics class would provide operations for graphical information, common base class holding graphical information.

```
class Comm_sat : public Satellite, public Displayed
{
  public:

    // ...
};
```

In addition to whatever operations defined specifically for `Comm_sat`, union of operations on `Satellite` and `Displayed` can be applied.

Use of multiple inheritance to "glue" 2 otherwise unrelated classes together as part of the implementation of 3rd. class is crude, effective, and relatively important.

"I generally prefer to have a single implementation hierarchy and (where needed) several abstract classes providing interfaces."
 - pp. 626 Stroustrup.

#### Ambiguity Resolution

pp. 627, Sec. 21.3.3 Ambiguity Resolution. Stroustrup.

Explicit disabmbiguation is messy, so it's usually best to resolve such problems by defining a new function in the derived class.

```
class Comm_sat : public Satellite, public Displayed
{
  public:

    Debug_info get_debug() // override Comm_sat::get_debug() and Displayed::get_debug()
    {
      Debug_info di1 = Satellite::get_debug();
      Debug_info di2 = Displayed::get_debug();
      return merge_info(di1, di2);
    }

};
```

A function declared in a derived class overrides *all* functions of the same name and type in its base classes.

Compiler recursively looks in its base classes.
  be careful of "infinite" recursive call insider implementation

### Repeated Use of a Base Class

cf. 21.3.4 Repeated Use of a Base Class, Stroustrup

When each class has only 1 direct base class, class hierarchy will be a tree, and a class can only occur once in the tree.

When class can have multiple base classes, class can appear multiple times in resulting hierarchy.

#### Virtual Base Classes

cf. 21.3.5. Virtual Base Classes, Stroustrup

Base class can be safely, conveniently, efficiently replicated if base class is an abstract class providing a pure interface, base class object holds no data of its own.
  - this simplest case offers best separation of interface and implementation concerns

What if base class hold data, and it was important that it shouldn't be replicated? Given this apparantly minor change to base class, we must change design of derived class; all parts of an object must shared a single copy of base class. Otherwise, we could get 2 parts of something derived from base class multiple times using different objects.
  - avoid replication by declaring base `virtual`: every `virtual` base of a derived class is represented by same (shared) object.

Why would someone want to use virtual base containing data? 3 ways for 2 classes in a class hierarchy to share data
1. Make data nonlocal (outside class as a global or namespace variable) // poor choice
2. put data in a base class
3. allocate object somewhere and give each of 2 classes a pointer.

1.  nonlocal data, usually a poor choice because we can't control what code accesses data and how, breaks encapsulation and locality
2. put data in base class, is simplest; however, every member of class hierarchy gets access.
3. sharing objects accessed through pointers; however ctors needs to set aside memory for that shared object, initialize it, provde pointers to shared object to objects needing access. That's roughly what ctors do to implement virtual base.

If you don't need sharing, you can do without virtual bases, and your code is often better and typically simpler for it.

##### Constructing Virtual Bases

cf. 21.3.5.1 Constructing Virtual Bases, Stroustrup

However complicated, language ensures that ctor of a virtual base is called exactly once.

Furthermore, ctor of a base (whether virtual or not) is called before its derived classes.

ctor of every virtual base is invoked (implicitly or explicitly) from ctor for complete object (ctor for most derived class).
  In particular, this ensures that virtual base is constructed exactly once even if it's mentioned in many places in the class hierarchy.

Logical problem with ctors doesn't exist for dtors; they're simply invoked in reverse order of ctor (Sec. 20.2.2). In particular, dtor for virtual base invoked exactly once.

cf. 21.3.5.2 Calling a Virtual Class Member Once Only

When defining functions for class with virtual base, programmer in general can't know whether base will be shared with other derived classes.
  - This can be problem when implementing a service that requires base class function to be called exactly once for each call of a derived function
  - Where needed, programmer can simulate scheme used for ctors by calling virtual base class function only from most derived class.

e.g.

```
class Window // base
{
  public:
    // basic stuff
    virtual void draw();
};

class Window_with_border : public virtual Window
{
  // border stuff
  protected:

    void own_draw();  // display the border

  public:

    void draw() override;
};

class Window_with_menu : public virtual Window
{
  // menu stuff
  protected:
    void own_draw(); // display the menu

  public:
    void draw() override;
};

class Clock : public Window_with_border, public Window_with_menu
{
  // clock stuff
  protected:

    void own_draw(); // display the clock face and hands

  public:

    void draw() override;
};

void Window_with_border::draw()
{
  Window::draw();
  own_draw(); // display the border
}

void Window_with_menu::draw()
{
  Window::draw();
  own_draw(); // display the menu
}

void Clock::draw()
{
  Window::draw();
  Window_with_border::own_draw();
  Window_with_menu::own_draw();
  own_draw(); // display the clock face and hands
}

```

cf. 21.3.6 Replicated vs. Virtual Bases, Stroustrup

In Sec. 21.2.2., Stroustrup made `Ival_box` classes abstract to reflect their role as pure interfaces. Doing that allowed Stroustrup to place all implementation details in specific implementation classes.

When using abstract class **(without any shared data)** as interface, we have a choice:
* replicate interface class (1 object per mention in the clas hierarchy)
* Make interface class `virtual` to share simple object among all classes in hierarchy that mention it.

`Ival_slider` as virtual base:

```
class Ival_slider;

class Popup_Ival_slider :
  public virtual Ival_slider
{
  /* ... */
};

class BB_ival_slider :
  public virtual Ival_slider,
  protected BBslider
{
  /* ... */
};


class BB_popup_ival_slider :
  public virtual Popup_Ival_slider,
  protected BB_ival_slider
{
  /* ... */
};
```

vs.

replicated `Ival_slider`

```
class Ival_slider;

class Popup_Ival_slider :
  public Ival_slider
{
  /* ... */
};

class BB_ival_slider :
  public Ival_slider,
  protected BBslider
{
  /* ... */
};


class BB_popup_ival_slider :
  public Popup_Ival_slider,
  protected BB_ival_slider
{
  /* ... */
};
```

Surprisingly there are no fundamental run-time or space advantages to 1 design over the other.

There are logical differences, though. In replicated `Ival_slider` design, `BB_popup_ival_slider` can't be implicitly converted to an `Ival_slider` (because that would be ambiguous):

```
void f(Ival_slider* p);

void g(BB_popup_ival_slider* p)
{
  f(p); // error: Popup_ival_slider::Ival_slider or BB_ival_slider::Ival_slider?
}
```

On the other hand, it's possible to construct plausible scenarios where sharing implied in virtual base design causes ambiguities for casts *from* the base class (Sec. 22.2). However, such ambiguities are easily dealt with.

How to choose between virtual base classes and replicated base classes for our (EY: pure) interfaces?
- When we do have a choice, we can take into account that (surprisingly) replicated base solution tends to lead to slightly smaller objects (because there's no need for data structures support sharing), and that we often get our interface objects from "virtual ctors" or "factor functions" (Sec. 21.2.4). e.g.

```
Popup_ival_slider* popup_slider_factor(args)
{
  // ...
  return new BB_popup_ival_slider(args);
}
```

No explicit conversion needed to get from an implementation (here `BB_popup_ival_slider`) to its direct interfaces (here, `Popup_ival_slider`). EY : So, the virtual base, in replicated, may be the problem???

##### Overriding Virtual Base Functions

A derived class can override virtual function of its direct or indirect virtual base class. In particular, 2 different classes might override different virtual functions from virtual base.

```
class Window
{
  // ...
  virtual void set_color(Color) = 0; // set background color
  virtual void prompt() = 0;
};

class Window_with_border : public virtual Window
{
  // ...
  void set_color(Color) override; // control background color
};

class Window_with_menu : public virtual Window
{
  // ...
  void prompt() override; // control user interactions
};

class My_window : public Window_with_menu, public Window_with_order
{
  // ...
};
```

What if different derived classes override the same function? 
This is allowed if and only if some overriding class is derived from every other class that overrides the function, i.e. one function must override all others.

```
class My_window: public Window_with_menu, public Window_with_border
{
  // ...
  void prompt() override; // don't leave user interactions to base
};
```
If 2 classes override base class function, but neither overrides other, class hierarchy is an error.
- reason is that no single function can be used to give a consistent meaning for all calls independently of which class they use as an interface; or no virtual function table can be constructed because a call to that function on the complete object would be ambiguous.
e.g. had `Radio` in Sec. 21.3.5 not declared `write()`, declarations of `write()` in `Receiver` and Transmitter would have caused an error when defining `Radio`.
As with `Radio`, such a conflict is resolved by adding an overriding function to most derived class.

###### mixin

**mixin** - a class that provides some - but not all - of the implementation for a virtual base class is often called a *mixin*.

## Advice on class hierarchy
cf. pp. 640 Class Hierarchies Ch. 21 21.4 Advice

1. Use `unique_ptr` or `shared_ptr` to avoid forgetting to `delete` objects created using `new` Sec. 21.2.1
4. Give abstract class a virtual destructor to ensure proper cleanup, Sec. 21.2.2.
6. Use abstract classes to support interface inheritance, Sec. 21.2.2
7. Use base classes with data members to support implementation inheritance; Sec. 21.2.2

# Run-Time Type Information

cf. Ch. 22 Run-Time Type Information. Stroustrup.

Recovering "lost" type of an object requires us to somehow ask the object to reveal its type.
- Any operation on an object requires us to have a pointer or reference of a suitable type for the object.
Consequently, most obvious and useful operation for inspecting type of an object at run time is a type conversion operation that returns a valid pointer if object is of expected type, and null pointer if it isn't. 
`dynamic_cast` operator does exactly that.

e.g. assume "the system" invokes `my_event_handler()` with a pointer to a `BBwindow`, where an activity has occurred. e.g. then might invoke my application code using `Ival_box`'s `do_something()`:

```
void my_event_handler(BBwindow* pw)
{
  if (auto pb = dynamic_cast<Ival_box*>(pw))  // does pw point to an Ival_box?
  {
    // ...
    int x = pb->get_value(); // use the Ival_box
    // ...
  }
  else
  {
    // ... oops! cope with unexpected event
  }
}
```
One way of explaining what's going on here is `dynamic_cast` translates from implementation-oriented language of user-interface system to language of the application.

It's important to note what's *not* mentioned in this example: the actual type of the object. Object will be a particular kind of `Ival_box`, say, an `Ival_slider`, implemented by a particular kind of `BBwindow`, say `BBslider`.

*downcast* - casting from a base class to derived class; base -> derived
*upcast* - cast from a derived class to base
*crosscast* - cast that goes from base to sibling class

`dynamic_cast` requires a ptr or reference to a polymorphic type in order to do downcast or crosscast.

**Require the pointer's type to be polymorphic**. 
- because it makes it easy to find a place to hold the necessary information about the object's type.
- a typical implementation will attach a "type information object" (Sec. 22.5) to an object by placing a pointer to the type information in the virtual function table for the object's class (Sec. 3.2.3).

If its type *is* known, we don't need to use `dynamic_cast`.

Target type of `dynamic_cast` need not be polymorphic. 
- This allows us to wrap a concrete type in polymorphic type, say,
- for transmission through an object I/O system (Sec. 22.2.4), and then "unwrap" the concrete type later.
e.g.

```
class IoObj // base class for object I/O system
{
  virtual IoObj* clone() = 0;
};

class IoDate : public Date, public IoObj
{};

void f(IoObj* pio)
{
  Date* pd = dynamic_cast<Date*>(pio);
  // ...
}
```

`dynamic_cast` to `void*` can be used to determine address of beginning of an object of polymorphic type. e.g.

```
void g(IvalBox* pb, Date* pd)
{
  void* pb2 = dynamic_cast<void*>(pb); // OK
  // void* pd2 = dynamic_cast<void*>(pd); // error: Date not polymorphic
}
```

Such casts are only useful for interaction with very low-level functions (only such functions deal with `void*`s). There's no `dynamic_cast` from `void*` (because there'd be no way of knowing where to find `vptr`; Sec. 22.2.3).









