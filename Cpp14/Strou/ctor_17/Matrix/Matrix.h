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
#include <utility> // std::move

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

 		Matrix(Matrix&&);												// move constructor
 		Matrix& operator=(Matrix&&);						// move assignment

 		~Matrix()
 		{
 			delete[] elements_;
 		}

 		//--------------------------------------------------------------------------
 		/// \details Having move operations affects idiom for returning large
 		/// objects from functions.
 		/// Matrix has a move ctor so that "return by value" is simple and
 		/// efficient, as well as "natural." Without move operations, we have
 		/// performance problems and must resort to workarounds.
    /// \ref pp. 517, Sec. 17.5.2, "Move" in Stroustrup, The C++ Programming
    /// Language, 4th Ed.
 		//--------------------------------------------------------------------------
 		template <class U>
 		friend Matrix<U> operator+(const Matrix<U>& a, const Matrix<U>& b)
 		// return-by-value
 		{
 			if (a.dim()[0] != b.dim()[0] || a.dim()[1] != b.dim()[1])
 			{
 				throw std::runtime_error("unequal Matrix sizes in +");
 			}

 			Matrix res {a.dim()[0], a.dim()[1]};
 			const auto n = a.size();
 			for (int i {0}; i != n; ++i)
 			{
 				res.elements_[i] = a.elements_[i] + b.elements_[i];
 			}

 			return res; 
 		}

    // Alternative implementation, pp. 533, "Passing Objects", Sec. 18.2.4 of
    // Stroustrup, The C++ Programming Language, 4th Ed.
    // Return-by-value
    /*
    Matrix operator+(const Matrix& a, const Matrix& b)
    {
      Matrix res {a};
      return res+=b;
    } */

 		//--------------------------------------------------------------------------
 		/// \ref pp. 533 18.2.4 Passing Objects. Ch. 18 Operator Overloading;
 		/// Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
 		/// \details operators that return 1 of their argument objects can - and
 		/// usually do - return a reference; particularly, common for operator
 		/// functions that are implemented as members.
 		//--------------------------------------------------------------------------
 		Matrix& operator+=(const Matrix& a) // return-by-value
 		{
 			if (dim_[0] != a.dim_[0] || dim_[1] != a.dim_[1])
 			{
 				throw std::runtime_error("bad Matrix += argument");
 			}

			T* p = elements_;
			T* q = a.elements_;
			T* end = p + dim_[0] * dim_[1];
			while (p != end)
			{
				*p++ += *q++;
			}

			return *this;
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

		//-------------------------------------------------------------------------
		/// \details How does the compiler know when it can use a move operation,
		/// rather than a copy operation? In general, we have to tell it by giving
		/// a rvalue reference argument.
		//-------------------------------------------------------------------------
		void swap(T& a, T& b) // "perfect swap" (almost)
		{
			T tmp = std::move(a);
			a = std::move(b);
			b = std::move(tmp);
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

//------------------------------------------------------------------------------
/// \brief Move constructor.
/// \details Move constructor simply takes representation from its source and
/// replace it with an empty Matrix.
//------------------------------------------------------------------------------
template <class T>
Matrix<T>::Matrix(Matrix&& a): // move constructor
	dim_{a.dim_},
	elements_{a.elements_} // grab a's representation
{
	a.dim_ = {0, 0};
	a.elements_ = nullptr; // clear a's representation
}

//------------------------------------------------------------------------------
/// \brief Move assignment.
/// \details Idea behind using a swap to implement a move assignment is that
/// the source do necessary cleanup work for us.
//------------------------------------------------------------------------------
template <class T>
Matrix<T>& Matrix<T>::operator=(Matrix&& a) // move assignment
{
	swap(dim_, a.dim_); // swap representations.
	swap(elements_, a.elements_);
	return *this;
}


} // namespace Matrices
} // namespace Modules

#endif // _MATRIX_H_
