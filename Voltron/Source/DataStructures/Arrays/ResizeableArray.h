//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating array as an Abstract Data Type.
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H
#define DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H

#include "Array.h"

#include <algorithm> // std::copy
#include <cstddef> // std::size_t
#include <initializer_list>
#include <stdexcept> // std::out_of_range, std::runtime_error
#include <string>

namespace DataStructures
{
namespace Arrays
{

//-----------------------------------------------------------------------------
/// \details new and delete operators allocate memory for objects from a pool
/// called free store.
//-----------------------------------------------------------------------------
template <typename T>
class ResizeableArray : Array<T>
{

  public:

    static constexpr int default_size_ {8};

    ResizeableArray():
      data_{new T[default_size_]},
      size_{default_size_},
      capacity_{default_size_}
    {}

    explicit ResizeableArray(const std::size_t N):
      data_{new T[N]{}},
      size_{N},
      capacity_{N}
    {
      if (N <= 0)
      {
        std::runtime_error(
          "ResizeableArray ctor: Invalid input N:" + std::to_string(N));
      }
    }

    explicit ResizeableArray(const std::initializer_list<T> list):
      data_{new T[list.size()]},
      size_{list.size()},
      capacity_{list.size()}
    {
      if (list.size() <= 0)
      {
        std::runtime_error("ResizeableArray ctor: Empty list:");
      }

      std::copy(data_, data_ + size_, list.begin());
    }

    explicit ResizeableArray(T* const data, const std::size_t N):
      data_{new T[N]},
      size_{N},
      capacity_{N}
    {
      if (N <= 0)
      {
        std::runtime_error(
          "ResizeableArray ctor: Invalid input N:" + std::to_string(N));
      }

      // Copy the data rather than copy the pointer, in fear of input pointer
      // data being deleted elsewhere.

      std::copy(data_, data_ + size_, data);
    }

    // Copies, Moves.

    // Copy ctor.
    ResizeableArray(const Resizeable& other):
      data_{new T[other.size()]},
      size_{other.size_},
      capacity_{other.capacity_}
    {
      std::copy(data_, data_ + size_, other.data_);
    }

    // Copy assignment.
    ResizeableArray& operator=(const ResizeableArray& other)
    {
      delete[] data_;
      data_ = new T[other.size()];
      size_ = other.size();
      capacity_ = other.capacity_;

      std::copy(data_, data_ + size_, other.data_);

      return *this;
    }

    // Move ctor.
    ResizeableArray(ResizeableArray&& other):
      data_{other.data_},
      size_{other.size()},
      capacity_{other.capacity_}
    {
      other.data_ = nullptr;

      // So not to invoke delete in dtor of other.
      other.size_ = 0;
    }

    // Move assignment.
    ResizeableArray& operator=(ResizeableArray&& other)
    {
      data_ = other.data_;
      other.size_ = 0;

      other.data_ = nullptr;
      return *this;
    }

    ~ResizeableArray()
    {
      // Release block of memory pointed by data_.
      // cf. https://www.softwaretestinghelp.com/new-delete-operators-in-cpp/
      // If delete data, data point to first element of array and this
      // statement will only delete first element of array. Using subscript
      // "[]", indicates variable whose memory is being freed is an array and
      // all memory allocated is to be freed.
      if (size() > 0)
      {
        delete[] data_;
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

		virtual const T& get_value(const int index) const
		{
			if (index >= size_ || index < 0)
			{
				throw std::out_of_range(
					"Out of Range, ResizeableArray: index input:" +
						std::to_string(index) +
						" size: " +
						std::to_string(size_));
			}

			return data_[index];
		}

		virtual void set_value(const int index, const T value)
		{
			if (index >= size_ || index < 0)
			{
				throw std::out_of_range(
					"Out of Range, ResizeableArray: index input:" +
						std::to_string(index) +
						" size: " +
						std::to_string(size_));
			}

			data_[index] = value;
		}

    virtual T& operator[](const size_t index)
    {
      return data_[index];
    }

    virtual const T& operator[](const size_t index) const
    {
      return data_[index];
    }

    // Returns an iterator to the beginning.
    constexpr T* begin()
    {
      return data_;
    }

    // Returns an iterator to the end.
    constexpr T* end()
    {
      return data_ + size_;
    }

    virtual size_t alignment_in_bytes() const
    {
      return alignof(ResizeableArray);
    }

    void append(T item)
    {
      ensure_extra_capacity();
      items_[size_] = item;
      size_++;
    }

	private:

    void ensure_extra_capacity()
    {
      // Then you have no actual space left.
      // Data Structures: Array vs ArrayList, HackerRank, Sep 20, 2016.
      // 7:01 for https://youtu.be/NLAzwv4D5iI
      // cf. https://stackoverflow.com/questions/37538/how-do-i-determine-the-size-of-my-array-in-c
      //
      // Wrong, No. Gets size of pointer for first, and size of element.
      //if (size_ == sizeof(data_) / sizeof(data_[0]))
      if (size_ == capacity_)
      {
        // Create a new array.
        T* new_copy {new T[size_ * 2]};

        // C++03 way
        std::copy(data_, data_ + size_, new_copy);
        //
        // C++11 way.
        //std::copy(std::begin(data_), std::end(data_), std::begin(new_copy));
        
        delete[] data_;

        // Items should now point to new_copy.
        data_ = new_copy;

        capacity_ = size_ * 2;
      }
    }

    T* data_;
    std::size_t size_;
    std::size_t capacity_;
};

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_RESIZEABLE_ARRAY_H