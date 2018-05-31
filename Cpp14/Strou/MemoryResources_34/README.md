# Memory and Resources; `unique_ptr`

cf. pp. 973, Ch. 34 **Memory and Resources** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.

## `unique_ptr` 

cf. pp. 987, Sec. 34.3.1. `unique_ptr` Ch. 34 **Memory and Resources** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.

`unique_ptr` provides semantics of strict ownership. 
* `unique_ptr` owns object to which it holds a pointer, i.e. it's `unique_ptr` obligation to destroy the object pointed to (if any) by its contained pointer. 
* `unique_ptr` can't be copied; can be moved
* `unique_ptr` stores a pointer and deletes object pointed to (if any) using associated deleter (if any) when it's itself destroyed (such as when a thread of control leaves `unique_ptr`'s scope (Sec. 17.2.2)

uses of `unique_ptr`:
* provide exception safety for dynamically allocated memory
* passing ownership of dynamically allocated memory to a function
* returning dynamically allocated memory from a function
* storing pointers in containers 

`unique_ptr<T,D>` (Sec. iso.20.7.1.2)
`cp` is the contained pointer 

| :--------- | :--------- | 
| `unique_ptr up {}` | Default ctor: `cp = nullptr`; constexpr; noexcept | 
| `unique_ptr up {p}` | `cp=p`; use default deleter; explicit; noexcept  |
| `unique_ptr up {p, del}` | `cp = p`; `del` is the deleter; noexcept    | 
| `unique_ptr up {up2}`    | Move ctor: `cp.p = up2.p`; `up2.p = nullptr`; noexcept |
| `up.~unique_ptr()` | Destructor: if `cp!=nullptr` invoke `cp`'s deleter | 
| `up = up2` | Move assignment: `up.reset(up2.cp);` `up2.cp = nullptr`; 
`up` gets `up2`'s deleter; `up`'s old object (if any) is deleted; noexcept |
|`up = nullptr` | `up.reset(nullptr)`; that is, delete `up`'s old object, if any |
| `bool b {up};` | Conversion to `bool: up.cp != nullptr`; explicit |

| :--------- | :--------- |
| `x=*up` | `x = up.cp;` for contained non-arrays only | 
| `x= up->m` | `x = up.cp->m`; for contained non-arrays only |
| `x = up[n]` | `x=up.cp[n]`; for contained arrays only |
| `x=up.get()` | `x = up.cp` | 
| `del=up.get_deleter()` | `del` is `up`'s deleter |
| `p=up.release()` | `p=up.cp`; `up.cp = nullptr` | 
| `up.reset(p)` | If `up.cp != nullptr` call deleter for `up.cp`; `up.cp = p` |
|`up.reset()` | `up.cp = pionter{}` (probably `nullptr`); 
  call the deleter for the old value of `up.cp` | 
|`up.swap(up2)` | Exchange `up` and `up2`'s values; noexcept |
| `up == up2` | `up.cp == up2.cp` |
| `up < up2` | `up.cp < up2.cp` |
| `up != up2` | `!(up == up2)` | 
| `up > up2` | `up2 < up` | 
| `up <= up2` | `!(up2 > up)` | 
| `up >= up2` | `!(up2 < up)` |
| `swap(up, up2)` | `up.swap(up2)` |

Note: `unique_ptr` doesn't offer a copy ctor or copy assignment. If you feel the need for copies, consider using `shared_ptr` (Sec. 34.3.2).

To avoid slicing (Sec. 17.5.1.4), a `Derived[]` isn't accepted as an argument to a `unique_ptr<Base[]>` even if `Base` is a public base of `Derived`. 

 

