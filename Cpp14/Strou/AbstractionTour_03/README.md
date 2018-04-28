# Abstraction Mechanisms, Classes, Class hierarchy Tour

cf. pp. 59, Ch. 3  **A Tour of C++:Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

## Concrete Types, Concrete classes

cf. pp. 59, Sec. 3.2.1 Concrete Types Ch. 3  **A Tour of C++:Abstraction Mechanisms** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

*concrete types* behave "just like built-in types". 

Defining characteristic of concrete type is that its representation is part of its definition. Allows implementations to be optimally efficient in time and space. Particularly,  
* places objects of concrete types on stack, in statically allocated memory, and in other objects (Sec. 6.4.2) 
* refer to objects directly (not just through pointers and references) 
* initialize objects immediately and completely (e.g. using constructors)
* copy objects (Sec. 3.3)

Representation can be private, and accessible only through member functions, but present. 
* Price to pay for having concrete types behave exactly like built-in types is if representation changes in any significant way, user must recompile. 

### Arithmetic Type; Complex number example 

Class definition itself contains only operations requiring access to representation. An industrial-strength `complex` (like the standard library one) is carefully implemented to do appropriate inlining. 

`const` specifiers on functions returning real and imaginary parts (`const` *after* function declaration) indicate these functions don't modify object for which they're called. 

### Container; RAII (Resource Acquisition Is Initialization) 
