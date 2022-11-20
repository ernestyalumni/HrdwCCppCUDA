#ifndef DATA_STRUCTURES_ARRAYS_DYNAMIC_ARRAY_H
#define DATA_STRUCTURES_ARRAYS_DYNAMIC_ARRAY_H

#include <algorithm> // std::copy, std::fill, std::max
#include <cassert>
#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Arrays
{

template <typename T>
void raw_destruct(T* array, const std::size_t size)
{
  for (std::size_t i {0}; i < size; ++i)
  {
    array[i].~T();
  }

  //----------------------------------------------------------------------------
  /// \url https://stackoverflow.com/questions/17344727/does-delete-call-destructors
  /// From Sec. 5.3.5 Delete of C++ standard, draft n3337, the delete-expression
  /// will invoke the dtor for object or elements of array being deleted. In
  /// case of array, elements will be destroyed in order of decreasing address
  /// (that is, in reverse order of completion of their ctor).
  /// \details We do not do the "array" delete, delete[]. We've already invoked
  /// the dtor for each element.
  //----------------------------------------------------------------------------
  //delete[] array;
  ::operator delete(array);
}

template <typename T>
T* raw_new(const std::size_t n = 1)
{
  //----------------------------------------------------------------------------
  /// \ref https://en.cppreference.com/w/cpp/memory/new/operator_new
  /// Class-specific allocation functions:
  /// void* T::operator new(std::size_t count);
  /// If defined, called by the usual single-object new-expressions if
  /// allocating object of type T.
  //----------------------------------------------------------------------------
  return (T*)::operator new(sizeof(T) * n);
}

template <typename T>
class DynamicArray
{
  public:

    inline static constexpr std::size_t default_capacity_ {8};

    DynamicArray():
      // cf. https://cplusplus.com/reference/new/operator%20new/
      // void* operator new (std::size_t size) throw
      // Allocates size bytes of storage, suitably aligned to represent any
      // object of that size, and returns a non-null ptr to first byte of this
      // block.
      data_{raw_new<T>(default_capacity_)},
      size_{0},
      capacity_{default_capacity_}
    {}

    DynamicArray(const std::size_t initial_size, const T& value = T{}):
      data_{raw_new<T>(std::max(initial_size, default_capacity_))},
      size_{0},
      capacity_{std::max(initial_size, default_capacity_)}
    {
      for (std::size_t i {0}; i < initial_size; ++i)
      {
        append(value);
      }
    }

    //--------------------------------------------------------------------------
    /// You *MUST* define at least copy semantics (Rule of 3) and preferably
    /// user define move semantics.
    /// \ref https://stackoverflow.com/questions/63413418/free-double-free-detected-in-tcache-2
    /// \details If copy ctor, copy assignment are not user-defined, written,
    /// they are generated with shallow semantics. Since class "owns" a ptr,
    /// namely, T* data_, then 2 instances of DynamicArray both own the same
    /// pointer, and when one is destroyed, it destroys the data pointed to it
    /// by the other.
    //--------------------------------------------------------------------------

    // Copy ctor.
    DynamicArray(const DynamicArray& other):
      data_{raw_new<T>(other.size())},
      size_{other.size_},
      capacity_{other.capacity_}
    {
      for (std::size_t i {0}; i < size_; ++i)
      {
        new(&data_[i])T(other.data_[i]);
      }
    }

    // Copy assignment.
    DynamicArray& operator=(const DynamicArray& other)
    {
      raw_destruct<T>(data_, size_);
      data_ = raw_new<T>(other.size());
      size_ = other.size();
      capacity_ = other.capacity_;

      for (std::size_t i {0}; i < size_; ++i)
      {
        new(&data_[i])T(other.data_[i]);
      }

      return *this;
    }

    // Move ctor.
    DynamicArray(DynamicArray&& other):
      data_{other.data_},
      size_{other.size_},
      capacity_{other.capacity_}
    {
      other.data_ = nullptr;

      other.size_ = 0;
    }

    // Move assignment.
    DynamicArray& operator=(DynamicArray&& other)
    {
      data_ = other.data_;
      other.size_ = 0;

      other.data_ = nullptr;
      return *this;
    }

    virtual ~DynamicArray()
    {
      raw_destruct<T>(data_, size_);
    }

    void append(const T& item)
    {
      if (size_ == capacity_)
      {
        resize_capacity();
      }

      //----------------------------------------------------------------------
      /// \url https://en.cppreference.com/w/cpp/language/new#Placement_new
      /// \details new (placement-params) (type) initializer
      /// Attempts to create an object of type, but provides additional
      /// arguments to the allocation function, placement-params are passed to
      /// the allocation function as additional arguments.
      //----------------------------------------------------------------------
      // Construct a T object, placing it directly into memory address of
      // &data_[size_], initialized to value at item. size_++ means to
      // increment afterwards.
      new(&data_[size_++])T(item);
    }

    void remove_last()
    {
      assert(size_ > 0);
      data_[--size_].~T();

      // When the size, i.e. number of elements, is less than 1/4 of the
      // capacity, we can shrink the capacity by half.
      if (capacity_ > default_capacity_ && size_ * 4 < capacity_)
      {
        resize_capacity();
      }
    }

    std::size_t size() const
    {
      return size_;
    }

    std::size_t capacity() const
    {
      return capacity_;
    }

    bool has_data() const
    {
      return data_ != nullptr;
    }

    T& operator[](const std::size_t i)
    {
      assert(i >= 0 && i < size_);
      return data_[i];
    }

    const T& operator[](const std::size_t i) const
    {
      assert(i >= 0 && i < size_);
      return data_[i];
    }

    // Returns an iterator to the beginning.
    constexpr T& begin()
    {
      return data_;
    }

    constexpr T* end()
    {
      return data_ + size_;
    }

  private:

    void resize_capacity()
    {
      // This is an old version; TODO: device to remove code comments or not.
      /*
      capacity_ = std::max(2 * size_, default_capacity_);

      // Create a new array.
      T* new_copy {new T[capacity_]};

      for (std::size_t i {0}; i < size_; ++i)
      {
        //----------------------------------------------------------------------
        /// \url https://en.cppreference.com/w/cpp/language/new#Placement_new
        /// \details new (placement-params) (type) initializer
        /// Attempts to create an object of type, but provides additional
        /// arguments to the allocation function, placement-params are passed to
        /// the allocation function as additional arguments.
        //----------------------------------------------------------------------
        // Construct a ITEM_T object, placing it directly into memory address of
        // &items_[i], initialized to value at old_items[i].
        new(&new_copy[i])T(data_[i]);
      }

      raw_destruct<T>(data_, size_);

      // Items should now point to new copy.
      data_ = new_copy;
      */

      T* old_data {data_};
      capacity_ = std::max(2 * size_, default_capacity_);
      // Allocate a new array of capacity = 2 x the size.
      data_ = raw_new<T>(capacity_);
      for (std::size_t i {0}; i < size_; ++i)
      {
        // Copy all the items into it (i.e. new array).
        new(&data_[i])T(old_data[i]);
      }
      raw_destruct<T>(old_data, size_);
    }

    T* data_;
    std::size_t size_;
    std::size_t capacity_;
};

template <typename T>
class PrimitiveDynamicArray
{
  public:

    inline static constexpr std::size_t default_capacity_ {8};

    PrimitiveDynamicArray():
      data_{new T[default_capacity_]},
      size_{0},
      capacity_{default_capacity_}
    {}

    PrimitiveDynamicArray(
      const std::size_t initial_size,
      const T value = T{}
      ):
      data_{new T[std::max(initial_size, default_capacity_)]},
      size_{0},
      capacity_{std::max(initial_size, default_capacity_)}
    {
      std::fill(data_, data_ + size_, value);
    }

    //--------------------------------------------------------------------------
    /// You *MUST* define at least copy semantics (Rule of 3) and preferably
    /// user define move semantics.
    /// \ref https://stackoverflow.com/questions/63413418/free-double-free-detected-in-tcache-2
    /// When copy ctor and copy assignment aren't user-defined, written
    /// yourself, they're generated with shallow copy semantics. Since this
    /// class "owns" a pointer, namely T* data_, if shallow copied, then 2
    /// instnaces of DynamicArray both own the same pointer, and when one is
    /// destroyed, it destroys data pointed to by the other.
    //--------------------------------------------------------------------------

    // Copy ctor.
    PrimitiveDynamicArray(const PrimitiveDynamicArray& other):
      data_{new T[other.size()]},
      size_{other.size_},
      capacity_{other.capacity_}
    {
      std::copy(other.data_, other.data_ + other.size(), data_);
    }

    // Copy assignment.
    PrimitiveDynamicArray& operator=(const PrimitiveDynamicArray& other)
    {
      delete[] data_;
      data_ = new T[other.size()];
      size_ = other.size();
      capacity_ = other.capacity_;

      std::copy(other.data_, other.data_ + other.size(), data_);

      return *this;
    }

    // Move ctor.
    PrimitiveDynamicArray(PrimitiveDynamicArray&& other):
      data_{other.data_},
      size_{other.size()},
      capacity_{other.capacity_}
    {
      other.data_ = nullptr;

      // So not to invoke delete effectively in the dtor of the other.
      other.size_ = 0;
    }

    // Move assignment.
    PrimitiveDynamicArray& operator=(PrimitiveDynamicArray&& other)
    {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.size_ = 0;

      other.data_ = nullptr;
      return *this;
    }

    virtual ~PrimitiveDynamicArray()
    {
      if (size() > 0 || data_ != nullptr)
      {
        delete[] data_;
      }
    }

    void append(T item)
    {
      if (size_ == capacity_)
      {
        resize_capacity();
      }
      data_[size_++] = item;
    }

    std::size_t size() const
    {
      return size_;
    }

    std::size_t capacity() const
    {
      return capacity_;
    }

    bool has_data() const
    {
      return data_ != nullptr;
    }

    T& operator[](const std::size_t i)
    {
      // Commented out because it could be the case where we set it in the
      // beginning.
      //assert(i < size_);
      return data_[i];
    }

    const T& operator[](const std::size_t i) const
    {
      // Commented out because it could be the case where we set it in the
      // beginning.
      //assert(i < size_);
      return data_[i];
    }

    // Returns an iterator to the beginning.
    constexpr T& begin()
    {
      return data_;
    }

    constexpr T* end()
    {
      return data_ + size_;
    }

  private:

    void resize_capacity()
    {
      T* old_data {data_};
      capacity_ = std::max(2 * size_, default_capacity_);
      data_ = new T[capacity_];

      // C++03 way.
      std::copy(old_data, old_data + size_, data_);

      delete[] old_data;
    }

    T* data_;
    std::size_t size_;
    std::size_t capacity_;
};

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_DYNAMIC_ARRAY_H