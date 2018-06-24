# Endianness, Floating Point representations

cf. [`endian.h` from Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/endian.3.html)

```
htobe16
htole16
be16toh
le16toh
htobe32
htole32
be32toh
le32toh
htobe64
be64toh
le64toh
```
convert values between host and big-/little-endian byte order.

```
#include <endian.h>

uint16_t htobe16(uint16_t host_16bits);

```


https://github.com/google/sensei/blob/master/sensei/util/endian.h
in 
https://github.com/google/sensei

### Value Representation; object representation

cf. [Scott Schurr - *Type Punning in C++17 - Avoiding Pun-defined Behavior*](https://github.com/CppCon/CppCon2017/blob/master/Presentations/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior%20-%20Scott%20Schurr%20-%20CppCon%202017.pdf)

*Object representation* of an object of type `T` is the sequence of `N` unsigned char objects taken up by the object of type `T`, where `N` equals `sizeof(T)`. 

*Value representation* of an object is the set of bits that hold the value of type `T`. 

For trivially copyable types, the value representation is a set of bits in the object representation that determines a *value*, which is 1 discrete element of an implementation-defined set of values. 

``` 
static_assert(std::is_trivially_copyable<T>::value)
```

#### Trivially Copyable Type Rules

* every copy and move ctor, copy and move assignment operator is trivial or deleted
* at least 1 copy and/or move is not deleted
* trivial non-deleted destructor
* no virtual members
* no virtual base classes
* every subobject must be trivially copyable.

### Implementation-Defined Behavior

`[defns.impl.defined]`

behavior, for a well-formed program construct and correct data, that depends on the implementation and that each implementation documents. 

#### Examples of Implementation-Defined Behavior

- number of bits in a byte; see `[intro.memory] Sec. 1` 
- which (if any) atomics are always lock free. cf. `[atomic.types.generic]` Sec. 4

### Unspecified Behavior

behavior, for a well-formed program construct and correct data, that depends on the implementation.

#### Unspecified Behavior Examples 

- how memory for an exception object is allocated. See `[except.throw]` Sec. 4

### Undefined Behavior 

Behavior for which this International Standard imposes no requirements

#### Undefined Behavior Examples

- consequences of overflow of a signed integer. See `[expr]` Sec. 4
- Any race condition. See `[intro.races]` Sec. 20
- Accessing non-active member of a union. Implied by `[class.union]` Sec. 1

Caution: Exhibited behavior can change based on compiler flags or compiler version.

## Casting away `const` 

### Casting `const` to non-`const`

cf. `[dcl.type.cv]` Sec. 4. 

Except that any class member declared `mutable` can be modified, any attempt to modify `const` object during its lifetime results in undefined behavior.

* casting away `const` is fine
* modifying a non-mutable `const` is undefined behavior.


#### Compiler Caching of `const` values

* if the optimizer sees a value is `const`, the optimizer is allowed to assume the value doesn't change. 

If you change a `const` value, you're violating the optimizer's assumption.

`std::launder` in C++17


### Pointers; Using an Invalid Pointer is Undefined

`[basic.stc.dynamic.safety]` Sec. 4, The effect of using an invalid pointer value (including passing it to a deallocation function) is undefined.

#### Pointer to Integer Conversions 




## Miscellaneous

`snprintf`

cf. http://www.cplusplus.com/reference/cstdio/snprintf/

``` 
int snprintf(char* s, size_t n, const char* format, ...);
``` 
**Write formatted output to sized buffer**.

Composes a string with the same text that would be printed if *format* was used on `printf`, but instead of being printed, the content is stored as a C *string* in the buffer pointed by `s` (taking `n` as the maximum buffer capacity to fill).

## References

[Scott Schurr - *Type Punning in C++17 - Avoiding Pun-defined Behavior*](https://github.com/CppCon/CppCon2017/blob/master/Presentations/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior%20-%20Scott%20Schurr%20-%20CppCon%202017.pdf)

https://benjaminjurke.com/content/articles/2015/loss-of-significance-in-floating-point-computations/

http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_2_Floating_point.pdf

* David Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. March 1991, Computing Surveys.

[Download link from waterloo.ca](https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/02Numerics/Double/paper.pdf)


