//------------------------------------------------------------------------------
/// \file ArrayList.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating list as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.2. An array-based list implementation.
/// \ref https://github.com/OpenDSA/OpenDSA/blob/master/SourceCode/C%2B%2B_Templates/Lists/AList.h
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LISTS_ARRAY_LIST_H
#define DATA_STRUCTURES_LISTS_ARRAY_LIST_H

#include "List.h"

#include <stdexcept> // std::out_of_range

namespace DataStructures
{
namespace Lists
{

//-----------------------------------------------------------------------------
/// \brief Array-based list implementation
//-----------------------------------------------------------------------------
template <typename T>
class ArrayList : public List<T>
{
	public:

		// Default size.
		static constexpr int default_size {10};

		// Constructors

    //--------------------------------------------------------------------------
    /// \brief Create a new list object with maximum size "size".
    //--------------------------------------------------------------------------
		ArrayList(const int size = default_size) :
			max_size_{size},
			list_size_{0},
			current_{0},
			list_array_{new T[max_size_]}
		{}

    //--------------------------------------------------------------------------
    /// \brief Destructor. Destructor to remove array.
    //--------------------------------------------------------------------------
		~ArrayList()
		{
			delete [] list_array_;
		}

    //--------------------------------------------------------------------------
    /// \brief Reinitialize the list.
    //--------------------------------------------------------------------------
		void clear()
		{
			// Remove the array.
			delete [] list_array_;

			// Reset the size.
			list_size_ = current_ = 0;


			// Recreate array.
			list_array_ = new T[max_size_];
		}

    //--------------------------------------------------------------------------
    /// \brief Insert "it" at current position.
    //--------------------------------------------------------------------------
		void insert(const T& it)
		{

			//assert(list_size_ < max_size_, "List capacity exceeded"); // Original

			if (list_size_ >= max_size_)
			{
				return;
			}

			// Shift elements up to make room
			//
			// Time complexity: worst case O(N).
			for (int i {list_size_}; i > current_; --i)
			{
				list_array_[i] = list_array_[i - 1];
			}

			list_array_[current_] = it;

			// Increment list size.
			++list_size_;
		}

    //--------------------------------------------------------------------------
    /// \brief Append "it".
    //--------------------------------------------------------------------------
		void append(const T& it)
		{
			//assert(list_size_ < max_size_, "List capacity exceeded"); // Original
			if (list_size_ >= max_size_)
			{
				return;
			}

			list_array_[list_size_++] = it;
		}

    //--------------------------------------------------------------------------
    /// \brief Remove and return the current element.
    //--------------------------------------------------------------------------
		T remove()
		{
			// originally,
			//assert((current_ >= 0) && (current_ < list_size_), "No element");

			if (current_ < 0 || current_ >= list_size_)
			{
				throw std::out_of_range("remove() in ArrayList has current of");
			}

			// Copy the element.
			T it {list_array_[current_]};

			// Shift them down.
			for (int i {current_}; i < list_size_ - 1; ++i)
			{
				list_array_[i] = list_array_[i + 1];
			}

			// Decrement size.
			list_size_--;

			return it;
		}

    //--------------------------------------------------------------------------
    /// \brief Reset position.
    //--------------------------------------------------------------------------
		void move_to_start()
		{
			current_ = 0;
		}

    //--------------------------------------------------------------------------
    /// \brief Set at end.
    //--------------------------------------------------------------------------
		void move_to_end()
		{
			current_ = list_size_;
		}

    //--------------------------------------------------------------------------
    /// \brief Back up.
    //--------------------------------------------------------------------------
		void previous()
		{
			if (current_ != 0)
			{
				current_--;
			}
		}

    //--------------------------------------------------------------------------
    /// \brief Next.
    //--------------------------------------------------------------------------
		void next()
		{
			if (current_ < list_size_)
			{
				current_++;
			}
		}

    //--------------------------------------------------------------------------
    /// \brief Return list size.
    //--------------------------------------------------------------------------
		int length() const
		{
			return list_size_;
		}

    //--------------------------------------------------------------------------
    /// \brief Return current position.
    //--------------------------------------------------------------------------
		int current_position() const
		{
			return current_;
		}

    //--------------------------------------------------------------------------
    /// \brief Set current list position to "position".
    //--------------------------------------------------------------------------
		void move_to_position(int position)
		{
			//assert(
			//	(position >= 0) && (position <= list_size_),
			//	"Position out of range"); // Original

			if (position < 0 || position > list_size_)
			{
				return;
			}

			current_ = position;
		}

    //--------------------------------------------------------------------------
    /// \brief Return current element
    //--------------------------------------------------------------------------
		const T& get_value() const
		{
			// Originally,
			//assert((current_ >= 0) && (current_ < list_size_), "No current element");
			if (current_ < 0 || current_ >= list_size_)
			{
				throw std::out_of_range("get_value() in ArrayList has current of ");
			}

			return list_array_[current_];			
		}

	private:

		// Maximum size of list.
		int max_size_;

		// Number of list items now.
		int list_size_;

		// Position of current element
		int current_;

		// Array holding list elements
		T* list_array_;
};

} // namespace Lists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LISTS_LISTS_H