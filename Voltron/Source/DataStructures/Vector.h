#ifndef DATA_STRUCTURES_VECTOR_H
#define DATA_STRUCTURES_VECTOR_H

#include "Utilities/ArithmeticType.h"
#include "Utilities/KedykUtilities.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace DataStructures
{

namespace Kedyk
{

//------------------------------------------------------------------------------
/// \ref 5.3 Vector. pp. 40. Implementing Useful Algorithms in C++. Kedyk.
/// \details An array of dynamic size is the simplest and most useful data
/// structure-even in cases where it's inefficient but not the bottleneck. It
/// models a collection such as a shopping list, but without any extra
/// properties that may be useful for efficient operations.
//------------------------------------------------------------------------------

template <typename ITEM_T>
class Vector : public Utilities::Kedyk::ArithmeticType<Vector<ITEM_T>>
{
  public:
    inline static constexpr std::size_t MIN_CAPACITY {8};

    explicit Vector():
      capacity_{MIN_CAPACITY},
      size_{0},
      items_{Utilities::Kedyk::raw_memory<ITEM_T>(capacity_)}
    // No default ITEM_T ctor needed.
    {}

    explicit Vector(
      const std::size_t initialize_size,
      const ITEM_T& value = ITEM_T{}
      ):
      capacity_{std::max(initialize_size, MIN_CAPACITY)},
      size_{0},
      items_{Utilities::Kedyk::raw_memory<ITEM_T>(capacity_)}
    {
      for (std::size_t i {0}; i < initialize_size; ++i)
      {
        append(value);
      }
    }

    ~Vector()
    {
      Utilities::Kedyk::raw_destruct(items_, size_);
    }

    // Setters and gettings.

    ITEM_T* get_array()
    {
      return items_;
    }

    const ITEM_T* get_array() const
    {
      return items_;
    }

    std::size_t get_size() const
    {
      return size_;
    }

    ITEM_T& operator[](const std::size_t i)
    {
      assert(i >= 0 && i < size_);
      return items_[i];
    }

    const ITEM_T& operator[](const std::size_t i) const
    {
      assert(i >= 0 && i < size_);
      return items_[i];
    }

    void append(const ITEM_T& item)
    {
      if (size_ >= capacity_)
      {
        resize();
      }

      new(&items_[size_++])ITEM_T(item);
    }

    void remove_last()
    {
      assert(size_ > 0);
      items_[--size_].~ITEM_T();

      // When the size, i.e. number of elements is less than 1/4 of the
      // capacity, we can 
      if (capacity_ > MIN_CAPACITY && size_ * 4 < capacity_)
      {
        resize();
      }
    }

  protected:

    //--------------------------------------------------------------------------
    /// A vector uses array doubling to create space for extra items when
    /// needed.
    /// This makes O(lg(n)) memory manager calls for n appends. Appending is
    /// worst-case O(n) and amortized O(1) because n/2 items must have been
    /// inserted since the last resizing.
    //--------------------------------------------------------------------------

    void resize()
    {
      ITEM_T* old_items = items_;
      capacity_ = std::max(2 * size_, MIN_CAPACITY);
      items_ = Utilities::Kedyk::raw_memory<ITEM_T>(capacity_);
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
        new(&items_[i])ITEM_T(old_items[i]);

        Utilities::Kedyk::raw_destruct(old_items, size_);
      }
    }

    const std::size_t get_capacity() const
    {
      return capacity_;
    }

  private:

    std::size_t capacity_;
    std::size_t size_;
    ITEM_T* items_;
};

} // namespace Kedyk

} // namespace DataStructures

#endif // DATA_STRUCTURES_VECTOR_H