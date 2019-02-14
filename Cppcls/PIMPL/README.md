cf. [Pointer To Implementation (pimpl), C++ Programming/Idioms, Wikibooks](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Idioms)

In C++, you have to declare member variables within class definition, which is then public, and that this is necessary so that an appropriate memory space is allocated means that abstraction of implementation is not possible in "all" classes.

However, at the cost of an extra pointer dereference, and function call, you can have this level of abstraction through the Pointer to Implementation.

# Alternatives to pImpl idiom

cf. [Pimpl, cppreference](https://en.cppreference.com/w/cpp/language/pimpl)

- inline implementation: private members and public members are members of the same class
- pure abstract class (OOP factory): users obtain an unique pointer to a lightweight or abstract base class, the implementation details are in the derived class that overrides its virtual member functions.

[PIMPL, Rule of Zero and Scott Meyers | Hot C++ Blog](http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html)

The idea is to use `std::unique_ptr` to handle ownership of the implementation. The only problem we have to solve is that special members of `std::unique_ptr` unable to handle (to deelte, to be more precise) incomplete types. Thus, they must be instantiated at the point where the implementation class is defined. So we force them to be instantiated in the source file, rather than in the header. To do so, we define `MeyerParser` class special members (dtor, move ctor, move assignment) in source file.




