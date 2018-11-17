# `std::map`

cf. [`std::map`, cppreference](https://en.cppreference.com/w/cpp/container/map)

```
<map>

template <
  class Key,
  class T,
  class Compare = std::less<Key>,
  class Allocator = std::allocator<std::pair<const Key, T>>
> class map;
```
Since C++17

```
namespace pmr
{

template <class Key, class T, class Compare = std::less<Key>>
using map = std::map<Key, T, Compare,
  std::pmr::polymorphic_allocator<std::pair<const Key, T>>>
}
```

```
<functional>

template <class T>
struct less; // since C++14
```
`std::less` - function object for performing comparisons.
```
std::less::operator()

constexpr bool operator()(const T& lhs, const T& rhs) const; // since C++14
```
Checks whether `lhs` is less than `rhs`.

```
<memory>

template <class T>
struct allocator
```
`std::allocator` class template is default *Allocator* used by all standard library containers. Allocates and deallocates uninitialized storage.


`std::map` is a sorted associative container, that contains key-value pairs with unique keys. Keys sorted by using comparison function `Compare`. 
  - Search, removal, insertion operations have logarithmic complexity.
  - Maps usually implemented as red-black trees.

**Member types** |
**Member type** | **Definition**
:------------- | :------------
key_type | Key
mapped_type | T
value_type | `std::pair<const Key, T>`
iterator | *BidirectionalIterator*
const_iterator | Constant *BidirectionalIterator*

**Member classes** |
:---------------- | ----------
`value_compare` | compares objects of type value_type

### `std::map::map` constructor

```
map();
explicit map(const Compare& comp,
  const Allocator& alloc = Allocator()); // 1

explicit map(const Allocator& alloc); // 1 since C++11
```

```
template<class InputIt>
map(InputIt first, InputIt last,
  const Compare& comp = Compare(),
  const Allocator& alloc = Allocator()); // 2

template<class InputIt>
map(InputIt first, IntputIt last,
  const Allocator& alloc); // C++14
```

```
map(const map& other); // 3
map(const map& other, const Allocator& alloc); // 3 C++11

map(map&& other); // 4 C++11
map(map&& other, const Allocator& alloc); // 4 C++11 
```

```
map(std::initializer_list<value_type> init,
  const Compare& comp = Compare(),
  const Allocator& alloc = Allocator()); // C++11

map(std::initializer_list<value_type> init,
  const Allocator&); // C++14
```



# `std::queue`

```
<queue>

template <
  class T,
  class Container = std::deque<T>
> class queue;
```
[`std::queue` cppreference](https://en.cppreference.com/w/cpp/container/queue)

`std::queue` class is a container adapter that gives programmer functionality of a queue, **FIFO** (first-in, first-out) data structure.

# `std::deque`

`std::deque` (double-ended queue) is an indexed sequence contianer that allows fast insertion and deletion at both its beginning and its end. 
  - additionally, insertion and deletion at either end of a deque never invalidates pointers or references to the rest of the elements.



