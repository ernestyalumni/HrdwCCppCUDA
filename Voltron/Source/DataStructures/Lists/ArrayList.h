//------------------------------------------------------------------------------
/// \file ArrayList.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating list as an Abstract Data Type.
/// \ref Data Structures and Algorithm Analysis in C++, 3rd. Ed.. Dr. Clifford
/// A. Shaffer. Fig. 4.2. An array-based list implementation.
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LISTS_ARRAY_LIST_H
#define DATA_STRUCTURES_LISTS_ARRAY_LIST_H

#include "List.h"

namespace DataStructures
{
namespace Lists
{

//-----------------------------------------------------------------------------
/// \brief Array-based list implementation
//-----------------------------------------------------------------------------
template <typename E>
class ArrayList : public List<E>
{
	public:

		ArrayList(const int size = default_size) :
			max_size_{size},
			list_size_{0},
			current_{0},
			list_array_{new E[max_size_]}
		{
		}

    //--------------------------------------------------------------------------
    /// \brief Destructor.
    //--------------------------------------------------------------------------
		~ArrayList()
		{
			delete [] list_array_;
		}

	private:

		// Maximum size of list.
		int max_size_;

		// Number of list items now.
		int list_size_;

		// Position of current element
		int current_;

		// Array holding list elements
		E* list_array_;

};

} // namespace Lists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LISTS_LISTS_H