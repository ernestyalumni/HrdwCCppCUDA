//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating array as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.1. The ADT for a list.
/// \ref https://github.com/OpenDSA/OpenDSA/blob/master/SourceCode/C%2B%2B_Templates/Lists/List.h
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H
#define DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H

#include "Array.h"

#include <cstddef> // std::size_t
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
class DynamicFixedSizeArray : Array<T>
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

		explicit DynamicFixedSizeArray(T* data, const size_t N):
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
			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}
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

			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}

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

		~DynamicFixedSizeArray()
		{
			if (size() > 0)
			{
				delete[] data_;
			}
		}

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
			return alignof(DynamicFixedSizeArray);
		}

	private:	

		T* data_;
		size_t size_;
};

template <std::size_t N, typename T, std::enable_if_t<(N > 0)>* = nullptr>
class FixedSizeArrayOnStack : Array<T>
{
	public:

		using size_t = std::size_t;

		FixedSizeArrayOnStack():
			data_{},
			size_{N}
		{}

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

		explicit FixedSizeArrayOnStack(T* data, const size_t size):
			data_{},
			size_{size}
		{
			if (N != size)
			{
				std::runtime_error(
					"FixedSizeArrayOnStack ctor: Invalid input size:" +
					std::to_string(size) +
					"N: " +
					std::to_string(N));
			}

			data_ = data;
		}

		// TODO: Fill this in.
		// Copies, Moves.
		/*
		// Copy ctor.
		DynamicFixedSizeArray(const DynamicFixedSizeArray& other):
			data_{new T[other.size()]},
			size_{other.size_}
		{
			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}
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

			for (size_t index {0}; index < size_; ++index)
			{
				data_[index] = other[index];
			}

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

		~DynamicFixedSizeArray()
		{
			if (size() > 0)
			{
				delete[] data_;
			}
		}

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
			return alignof(DynamicFixedSizeArray);
		}
		*/

	private:	

		T data_[N];
		size_t size_;
};



} // namespace Arrays
} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAYS_FIXED_SIZE_ARRAYS_H