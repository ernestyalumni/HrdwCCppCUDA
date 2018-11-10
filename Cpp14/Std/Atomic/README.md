# `atomic` `std::atomic`

[`std::atomic`, cppreference](https://en.cppreference.com/w/cpp/atomic/atomic)

```
<atomic>

template <class T>
struct atomic;

template <class T>
struct atomic<T*>;

<memory>
template <class T>
struct atomic <std::shared_ptr<T>>;

template <class T>
struct atomic<std::weak_ptr<T>>;
```

`std::atomic` template defines atomic type. If 1 thread writes to an atomic object while another thread reads from it, behavior well-defined (see [memory model](https://en.cppreference.com/w/cpp/atomic/atomic) for details on data races)

`std::atomic` is neither copyable nor movable.

