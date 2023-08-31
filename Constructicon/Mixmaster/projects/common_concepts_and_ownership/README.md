## Ownership 

https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html

### The Stack and the Heap

Stack is the stack.
Heap - memory allocator finds empty spot in the heap that's big enough, marks it as being in use, and returns a pointer, which is address of that location.

### Ownership Rules

* Each value in Rust has an *owner*.
* There can be only 1 owner at a time.
* When owner goes out of scope, value will be dropped.

### Variable Scope

* When `s` comes into scope, it's valid.
* It remains valid until to goes *out* of scope.