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

  delete[] array;
}

template <typename T>
class DynamicArray
{
  public:

    inline static constexpr std::size_t default_capacity_ {8};

    DynamicArray():
      data_{new T[default_capacity_]},
      size_{0},
      capacity_{default_capacity_}
    {}

    DynamicArray(const std::size_t initial_size, const T& value = T{}):
      data_{new T[std::max(initial_size, default_capacity_)]},
      size_{0},
      capacity_{std::max(initial_size, default_capacity_)}
    {
      for (std::size_t i {0}; i < initial_size; ++i)
      {
        append(value);
      }
    }

    virtual ~DynamicArray()
    {
      //if (size_ > 0 || data_ != nullptr)
      //{
      raw_destruct<T>(data_, size_);
      //}
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
      data_ = new T[capacity_];
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
      const T value
      ):
      data_{new T[std::max(initial_size, default_capacity_)]},
      size_{0},
      capacity_{std::max(initial_size, default_capacity_)}
    {
      std::fill(data_, data_ + size_, value);
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