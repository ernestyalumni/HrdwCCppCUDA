//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating array as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.1. The ADT for a list.
/// \ref https://github.com/OpenDSA/OpenDSA/blob/master/SourceCode/C%2B%2B_Templates/Lists/List.h
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H
#define DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H

#include "BaseArray.h"

#include <algorithm> // std::copy, std::fill
#include <cassert>
#include <cstddef> // std::size_t
#include <initializer_list>
#include <stdexcept> // std::out_of_range, std::runtime_error
#include <string>
#include <type_traits> // std::enable_if_t

namespace DataStructures
{
namespace Arrays
{

//-----------------------------------------------------------------------------
/// \brief Dynamically-allocated fixed size array.
/// \details C-style array underneath.
//-----------------------------------------------------------------------------
template <typename T>
class DynamicFixedSizeArray : BaseArray<T>
{
	public:

		using size_t = std::size_t;

		DynamicFixedSizeArray() = delete;

		explicit DynamicFixedSizeArray(const size_t N):
			data_{new T[N]{}},
			size_{N}
		{
			if (N <= 0)
			{
				std::runtime_error(
					"DynamicFixedSizeArray ctor: Invalid input N:" + std::to_string(N));
			}
		}

		explicit DynamicFixedSizeArray(const std::initializer_list<T> list):
			data_{new T[list.size()]},
			size_{list.size()}
		{
			if (list.size() <= 0)
			{
				std::runtime_error("DynamicFixedSizeArray ctor: Empty list:");
			}

			T* data_ptr {data_};

			for (auto x : list)
			{
				*data_ptr = x;
				++data_ptr;
			}
		}

		explicit DynamicFixedSizeArray(T* const data, const size_t N):
			data_{new T[N]},
			size_{N}
		{
			if (N <= 0)
			{
				std::runtime_error(
					"DynamicFixedSizeArray ctor: Invalid input N:" + std::to_string(N));
			}

			// Copy the data rather than copy the pointer, in fear of input pointer
			// data being deleted elsewhere.

			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = data[index];
			}
		}

		// Copies, Moves.

		// Copy ctor.
		DynamicFixedSizeArray(const DynamicFixedSizeArray& other):
			data_{new T[other.size()]},
			size_{other.size_}
		{
      std::copy(other.data_, other.data_ + other.size_, data_);
		}

		// Copy assignment.
		DynamicFixedSizeArray& operator=(const DynamicFixedSizeArray& other)
		{
			if (this->size() != other.size())
			{
				std::runtime_error(
					"Mismatching sizes - this size:" +
					std::to_string(this->size()) +
					"other size: " +
					std::to_string(other.size()));
			}

      std::copy(other.data_, other.data_ + other.data_ + other.size_, data_);

			return *this;
		}

		// Move ctor.
		DynamicFixedSizeArray(DynamicFixedSizeArray&& other):
			data_{other.data_},
			size_{other.size_}
		{
			other.data_ = nullptr;

			// So not to invoke delete in dtor of other.
			other.size_ = 0;
		}

		// Move assignment.
		DynamicFixedSizeArray& operator=(DynamicFixedSizeArray&& other)
		{
			if (this->size() != other.size())
			{
				std::runtime_error(
					"Mismatching sizes - this size:" +
					std::to_string(this->size()) +
					"other size: " +
					std::to_string(other.size()));
			}

			data_ = other.data_;
			other.size_ = 0;

			other.data_ = nullptr;
			return *this;
		}

		virtual ~DynamicFixedSizeArray()
		{
			if (size() > 0 || data_ != nullptr)
			{
				delete[] data_;
			}
		}

		virtual const T& get_value(const int index) const
		{
			if (index >= size_ || index < 0)
			{
				throw std::out_of_range(
					"Out of Range, DynamicFixedSizeArray: index input:" +
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
					"Out of Range, DynamicFixedSizeArray: index input:" +
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

		virtual size_t size() const
		{
			return size_;
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
			return alignof(DynamicFixedSizeArray);
		}

	private:	

		T* data_;
		size_t size_;
};

template <typename T, std::size_t N, std::enable_if_t<(N > 0)>* = nullptr>
class FixedSizeArrayOnStack : BaseArray<T>
{
	public:

		using size_t = std::size_t;

		FixedSizeArrayOnStack():
			data_{},
			size_{N}
		{}

		/*
		explicit FixedSizeArrayOnStack(const std::initializer_list<T> list):
			data_{},
			size_{list.size()}
		{
			if (list.size() != N)
			{
				std::runtime_error(
					"FixedSizeArrayOnStack ctor; mismatching sizes; list size: " +
					std::to_string(list.size()) +
					"N: " +
					std::to_string(N));
			}

			T* data_ptr {data_};

			for (auto x : list)
			{
				*data_ptr = x;
				++data_ptr;
			}
		}
		*/

		FixedSizeArrayOnStack(const std::initializer_list<T> list):
			data_{},
			size_{N}
		{
			if (list.size() <= N)
			{
				std::copy(list.begin(), list.end(), data_);
			}
			else
			{
				std::copy(list.begin(), list.begin() + N, data_);
			}
		}

		explicit FixedSizeArrayOnStack(T* data, const size_t size):
			data_{},
			size_{N}
		{
			if (N <= size)
			{
				std::copy(data, data + N, data_);
			}
			else
			{
				std::copy(data, data + size, data_);
			}
		}

		// Copies, Moves.
		// Copy ctor.
		FixedSizeArrayOnStack(const FixedSizeArrayOnStack& other):
			data_{},
			size_{other.size_}
		{
			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}
		}

		// Copy assignment.
		FixedSizeArrayOnStack& operator=(const FixedSizeArrayOnStack& other)
		{
			if (this->size() != other.size())
			{
				std::runtime_error(
					"Mismatching sizes - this size:" +
					std::to_string(this->size()) +
					"other size: " +
					std::to_string(other.size()));
			}

			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}

			return *this;
		}

		// Move ctor.
		FixedSizeArrayOnStack(FixedSizeArrayOnStack&& other):
			data_{other.data_},
			size_{other.size_}
		{
			other.data_ = nullptr;

			other.size_ = 0;
		}

		// Move assignment.
		FixedSizeArrayOnStack& operator=(FixedSizeArrayOnStack&& other)
		{
			if (this->size() != other.size())
			{
				std::runtime_error(
					"Mismatching sizes - this size:" +
					std::to_string(this->size()) +
					"other size: " +
					std::to_string(other.size()));
			}

			data_ = other.data_;
			other.size_ = 0;

			other.data_ = nullptr;
			return *this;
		}

		~FixedSizeArrayOnStack()
		{}

		virtual const T& get_value(const int index) const
		{
			if (index >= size_ || index > 0)
			{
				throw std::out_of_range(
					"Out of Range, DynamicFixedSizeArray: index input:" +
						std::to_string(index) +
						" size: " +
						std::to_string(size_));
			}

			return data_[index];
		}

		virtual void set_value(const int index, const T value)
		{
			if (index >= size_ || index > 0)
			{
				throw std::out_of_range(
					"Out of Range, DynamicFixedSizeArray: index input:" +
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

		virtual size_t size() const
		{
			return size_;
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
			return alignof(FixedSizeArrayOnStack);
		}

	private:	

		T data_[N];
		size_t size_;
};

namespace CRTP
{

template <typename T, std::size_t N>
class DynamicFixedSizeNArray : public BaseArray<T, DynamicFixedSizeNArray<T, N>>
{
	public:

		using size_t = std::size_t;

    DynamicFixedSizeNArray():
      data_{new T[N]},
      size_{N}
    {}

    DynamicFixedSizeNArray(const std::initializer_list<T> list):
      data_{new T[list.size()]},
      size_{N}
    {
      assert(list.size() > 0);

      std::copy(list.begin(), list.end(), data_);
    }

    DynamicFixedSizeNArray(T* const data):
      data_{new T[N]},
      size_{N}
    {
      // Copy the data rather than copy the pointer, in fear of input pointer
      // data being deleted elsewhere.

      std::copy(data, data + size_, data_);
    }

    // Copy ctor.
    DynamicFixedSizeNArray(const DynamicFixedSizeNArray& other):
      data_{new T[other.size()]},
      size_{other.size_}
    {
      std::copy(other.data_, other.data_ + other.size(), data_);
    }

    // Copy assignment.
    DynamicFixedSizeNArray& operator=(const DynamicFixedSizeNArray& other)
    {
      delete[] data_;
      data_ = new T[other.size()];
      size_ = other.size();

      std::copy(other.data_, other.data_ + other.size(), data_);

      return *this;
    }

		// Move ctor.
		DynamicFixedSizeNArray(DynamicFixedSizeNArray&& other):
			data_{other.data_},
			size_{other.size_}
		{
			other.data_ = nullptr;

			// So not to invoke delete in dtor of other.
			other.size_ = 0;
		}

		// Move assignment.
		DynamicFixedSizeNArray& operator=(DynamicFixedSizeNArray&& other)
		{
			if (this->size() != other.size())
			{
				std::runtime_error(
					"Mismatching sizes - this size:" +
					std::to_string(this->size()) +
					"other size: " +
					std::to_string(other.size()));
			}

			data_ = other.data_;
			other.size_ = 0;

			other.data_ = nullptr;
			return *this;
		}		

		~DynamicFixedSizeNArray()
		{
			if (size() > 0)
			{
				delete[] data_;
			}
		}

		const T& get_value(const size_t index) const
		{
			if (index >= size_ || index > 0)
			{
				throw std::out_of_range(
					"Out of Range, DynamicFixedSizeArray: index input:" +
						std::to_string(index) +
						" size: " +
						std::to_string(size_));
			}

			return data_[index];
		}

		void set_value(const size_t index, const T value)
		{
			if (index >= size_ || index > 0)
			{
				throw std::out_of_range(
					"Out of Range, DynamicFixedSizeArray: index input:" +
						std::to_string(index) +
						" size: " +
						std::to_string(size_));
			}

			data_[index] = value;
		}

		T& operator[](const size_t index)
		{
			return data_[index];
		}

		const T& operator[](const size_t index) const
		{
			return data_[index];
		}

		size_t size() const
		{
			return size_;
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

	private:

		T* data_;
		size_t size_;
};

} // namespace CRTP

} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H