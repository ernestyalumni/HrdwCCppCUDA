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

		using std::size_t;

		DynamicFixedSizeArray(const size_t size);

		// Copies, Moves.

	private:	

};
