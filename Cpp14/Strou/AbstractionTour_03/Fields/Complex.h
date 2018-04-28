/**
 * @file   : Complex.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Complex numbers (double type) as Concrete class 
 * @details Concrete class - defining property is its representation is its 
 * 	 definition
 * @ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
 * 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
 * @details : 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 FileOpen.cpp FileOpen_main.cpp -o FileOpen_main
 * */
#ifndef _COMPLEX_H_
#define _COMPLEX_H_

#include <cmath> // std::sqrt

namespace Fields
{

class Complex
{
	public:
		Complex(double r, double i):	// construct complex from 2 scalars
			re_{r}, im_{i}
		{}
		Complex(double r):						// construct complex from 1 scalar
			re_{r}, im_{0}
		{}
		Complex():										// default complex: {0,0}
			re_{0}, im_{0}		
		{}		

		// Accessors
		double real() const 
		{
			return re_;
		}

		double imag() const
		{
			return im_;
		}

		// Setters
		void real(double d)
		{
			re_ = d;
		}

		void imag(double d)
		{
			im_ = d;
		}

		// unary arithmetic
		Complex& operator+=(Complex z) 
		{
			re_ += z.real();
			im_ += z.imag();
			return *this;
		}

		Complex& operator-=(Complex z)
		{
			re_ -= z.real();
			im_ -= z.imag();
			return *this;
		}

		Complex& operator*=(Complex); 		// defined out-of-class somewhere
		Complex& operator/=(Complex);

		/// Complex Conjugation
		//--------------------------------------------------------------------------
		/// \brief Return the complex conjugate of this complex number.
		//--------------------------------------------------------------------------
		Complex conjugate()
		{
			return {re_, -im_};
		}

		//--------------------------------------------------------------------------
		/// \brief Conjugate this complex number itself.
		//--------------------------------------------------------------------------
		void conjugation()
		{
			im_ = -im_;
		}

	private:
		double im_;
		double re_;		

}; // class Complex

Complex operator+(Complex a, Complex b);

Complex operator-(Complex a, Complex b);

Complex operator-(Complex a);

Complex operator*(Complex a, Complex b);

Complex operator/(Complex a, Complex b);

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
bool operator==(Complex a, Complex b);		// equal

bool operator!=(Complex a, Complex b); 	// not equal 

// originally from Stroustrup, pp. 61
//Complex sqrt(Complex);
//------------------------------------------------------------------------------
// \ref https://en.wikipedia.org/wiki/Complex_number#Elementary_operations
/// \details r = |z| = |x + yi| = \sqrt{x^2 + y^2}
//------------------------------------------------------------------------------
double modulus(Complex z);

double modulusSquared(Complex z);

} // namespace Fields

#endif // _COMPLEX_H_
