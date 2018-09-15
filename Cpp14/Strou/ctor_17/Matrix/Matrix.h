//------------------------------------------------------------------------------
/// \file   : Matrix.h
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : "Resource Acquisition Is Initialization"
/// \ref    : 17.5.1 Copy Ch. 17 Construction, Cleanup, Copy, and Move; 
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
/// \detail  
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <algorithm> // std::copy
#include <array>
#include <cstddef> // std::size_t
#include <initializer_list> // std::initializer_list
#include <iostream>
#include <memory> // std::uninitialized_copy
#include <stdexcept> // std::runtime_error

namespace Modules
{
namespace Matrices
{

template<class T>
class Matrix
{
 	public:

 		explicit Matrix(int d1, int d2):
 			dim_{d1, d2}, elements_{new T[d1*d2]}	// simplified (no error handling)
 		{}

 		explicit Matrix(
 			const int d1,
 			const int d2,
 			std::initializer_list<T> elements
 			):
 			Matrix {d1, d2}
 		{
 			std::copy(elements.begin(), elements.end(), &elements_[0]);
 		}


 		Matrix(const Matrix&);									// copy constructor
 		Matrix& operator=(const Matrix&);				// copy assignment

 		Matrix(Matrix&&) = default;												// move constructor
 		Matrix& operator=(Matrix&&) = default;						// move assignment

 		~Matrix()
 		{
 			delete[] elements_;
 		}

 	//----------------------------------------------------------------------------
 	/// Accessors 
 	//----------------------------------------------------------------------------
 		int size() const
 		{
 			return dim_[0] * dim_[1];
 		}

 		const std::array<int, 2> dim() const
 		{
 			return dim_;
 		}

 		const T* elements() const
 		{
 			return elements_;
 		}

 		// row-major order, 0-indexed
 		T element(const int i, const int j) const
 		{
 			if (!valid_indices(i, j))
 			{
 				std::runtime_error("indices out of bounds");
 			}

 			return elements_[i*dim_[0] + j];
 		}

		void set_element(const int i, const int j, const T element)
		{
 			if (!valid_indices(i, j))
 			{
 				std::runtime_error("indices out of bounds");
 			}

 			elements_[i*dim_[0] + j] = element;
		} 		

    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<U> & A);

	private:

		bool valid_indices(const int i, const int j) const
		{
			return (i < dim_[0] && j < dim_[1]);
		}

		std::array<int, 2> dim_;			// 2 dimensions
		T* elements_;									// point to dim[0]*dim[1] elements of type T
};


//------------------------------------------------------------------------------
/// \details Note that default copy (copy the members) would be disastrously
/// wrong; Matrix elements wouldn't be copied, Matrix copy would have a pointer
/// to the same elements as the source, and Matrix destructor would delete the
/// (shared) elements twice (Sec. 3.3.1)
/// Copy ctor initializes uninitialized memory.
/// \ref https://en.cppreference.com/w/cpp/memory/uninitialized_copy
/// std::uninitialized_copy copies elements from the range [first, last) to an
/// uninitialized memory area beginning at d_first as if by for loop:
/// for (; first != last; ++d_first, (void) ++ first)
/// 	::new (static_cast<void*>(std::addressof(*d_first)))
/// 		typename std::iterator_traits<ForwardIt>::value_type(*first)
// error: invalid use of template-name ‘Modules::Matrices::Matrix’ without an
// argument list if you follow Stroustrup's example.
//------------------------------------------------------------------------------
template<class T>
Matrix<T>::Matrix(const Matrix& m):	// copy constructor
	dim_{m.dim()},
	elements_{new T[m.size()]}
{
	std::uninitialized_copy(m.elements_, m.elements_ + m.size(), elements_);
}

//------------------------------------------------------------------------------
/// \details Copy assignment operator must correctly deal with an object that
/// has already been constructed and may own resources.
/// \ref https://en.cppreference.com/w/cpp/algorithm/copy
/// std::copy copies all elements in the range, defined by [first, last), to
/// another range beginning at d_first.
//------------------------------------------------------------------------------
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix& m)		// copy assignment
{
	if (dim[0] != m.dim()[0] || dim_[1] != m.dim_[1])
	{
		throw std::runtime_error("bad size in Matrix = ");
	}
	std::copy(m.elements_(), m.elements_() + m.size(), elements_); // copy elements
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& A)
{
	for (int i {0}; i < A.dim_[0]; ++i)
	{
		for (int j {0}; j < A.dim_[1]; ++j)
		{
			os << A.element(i, j) << ' ';
		}
		os << '\n';
	}
	os << '\n';

  return os;
}



} // namespace Matrices
} // namespace Modules

#endif // _MATRIX_H_
