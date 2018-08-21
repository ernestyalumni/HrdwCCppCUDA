# Idioms

## Singletons

### `static` is now thread-safe in C++11. 

[Storage class specifiers; Static local variables](https://en.cppreference.com/w/cpp/language/storage_duration#Static_local_variables)

`static` specified variables, declared at block scope, have static storage duration, but are initialized the first time control passes through their declaration (unless their initialization is zero- or constant-initialization, which can be performed before block is first entered.) 

cf. https://herbsutter.com/2013/09/09/visual-studio-2013-rc-is-now-available/ Thread-safe function local static initialization (aka “magic statics”).

[What is the lifetime of a static variable in a C++ function? `stackoverflow`](https://stackoverflow.com/questions/246564/what-is-the-lifetime-of-a-static-variable-in-a-c-function)


# References

Andrei Alexandrescu. **Modern C++ Design: Generic Programming and Design Patterns Applied.** Addison Wesley. Feb. 1, 2001. ISBN: 0-201-70431-5.  
