# `std::forward`

cf. https://en.cppreference.com/w/cpp/utility/forward

```
<utility>

template <class T>
T&& forward(typename std::remove_reference<T>::type& t) noexcept;

template <class T>
constexpr T&& forward(std::remove_reference_t<T>& t) noexcept; // since C++14
```

Forwards lvalues as either lvalues or as rvalues, depending on T

```
template <class T>
constexpr T&& forward(std::remove_reference_t<T>&& t) noexcept // since C++14
```
Forwards rvalues as rvalues and prohibits forwarding of rvalues as lvalues.

# `std::is_invocable`

```
<type_traits>

template <class Fn, class... ArgTypes>
struct is_invocable; 
```
Determines whether `Fn` can be invoked with arguments `ArgTypes...` Formally, determines whether `INVOKE(declval<Fn>(), declval<ArgTypes>()...)` is well formed when treated as an unevaluated operand, where `INVOKE` is the operation defined in `Callable`.

```
template <class R, class Fn, class.. ArgTypes>
struct is_invocable_r
```
Determines whether `Fn` can be invoked with the arguments `ArgTypes...` to yield a result that's convertible to `R`.

Invocable is in C++17. For before, consider implementing:

cf. https://stackoverflow.com/questions/51187974/can-stdis-invocable-be-emulated-within-c11

template <typename F, typename... Args>
struct is_invocable :
    std::is_constructible<
        std::function<void(Args ...)>,
        std::reference_wrapper<typename std::remove_reference<F>::type>
    >
{
};

template <typename R, typename F, typename... Args>
struct is_invocable_r :
    std::is_constructible<
        std::function<R(Args ...)>,
        std::reference_wrapper<typename std::remove_reference<F>::type>
    >
{
};