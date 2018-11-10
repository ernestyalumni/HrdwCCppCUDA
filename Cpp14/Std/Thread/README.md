# Thread (Thread support library)

## `std::mutex`

```
<mutex>
class mutex
```

`mutex` class is a synchronization primitive that can be used to protect shared data from being simultaneously accessed by multiple threads.

`mutex` offers exclusive, non-recursive ownership semantics.
* calling thread *owns* a `mutex` from time that it successfully calls either `lock` or `try_lock` until it calls `unlock`
* When a thread owns a `mutex`, all other threads will block (for calls to `lock`) or receive a `false` return value (for `try_lock`), if they attempt to claim ownership of `mutex`
* calling thread must not own `mutex` prior to calling `lock` or `try_lock`

Behavior of program undefined if `mutex` destroyed while still owned by any threads, or thread terminates while owning a `mutex`.

### `std::mutex::mutex` (ctor)

```
constexpr mutex() noexcept; \\ ctor, mutex in unlocked state after ctor completes

mutex(const mutex&) = delete; \\ copy ctor deleted
```

Because default ctor is `constexpr`, static mutexes are initialized as part of static, non-local initialization, before any dynamic non-local initialization begins.
  * makes it safe to lock a mutex in a ctor of any static object.

  

