cf. pp. 298 Ch. 11 Select Operations

# 7.1 Introduction

In C++ (most) objects **"have identity"** - reside at specific memory address, object can be accessed if you know its address and type.

Pointers and references hold and use addresses.

# 7.2 Pointers 

For type `T`, `T*` is type "pointer to `T`", i.e. variable of type `T*` can hold address of an object of type `T`.

Implementation of ptrs is intended to map directly to addressing mechanisms of machine on which program runs. Most machines can address a byte.
Those that can't tend to have hardware to extract bytes from words. 
On the other hand, few machines can directly address an individual bit.
Consequently, smallest object that can be independently allocated and pointer to using ptr type is a `char`. Note `bool` is at least `char` space.
To store smaller values more compactly, you can use bitwise logical operations (11.1.1), bit-fields in structures (8.2.7), bitset (34.2.2).

## 7.2.1 `void*`

In low-level code, occasionally need to store or pass along address of memory location without actually knowing what type of object is stored there. 

Primary use for `void*` is for passing ptrs to functions that aren't allowed to make assumptions about type of object, and for returning untyped objects from functions. 

To use such an object, we must use explicit type conversion.

Where used for optimization `void*` can be hidden behind type-safe interface (27.3.1)

## 7.2.2 `nullptr`

`nullptr` - ptr that doesn't point to an object

