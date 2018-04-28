/**
 * @file   : Complex.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Complex numbers (double type) as Concrete class 
 * @ref    : 3.2.1.1 An Arithmetic Type, Ch. 3 A Tour of C++: Abstraction 
 * 	Mechanisms. Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
 * @details : Using RAII for Concrete classes. 
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
#include "Complex.h"

#include <cassert> 	// assert

namespace Fields
{

//------------------------------------------------------------------------------
/// \brief Complex multiplication and division
//------------------------------------------------------------------------------
/// unary multiplication and division

//------------------------------------------------------------------------------
/// \details z*w = (a+bi)*(c+di) = (a*c - b*d) + i(a*d + b*c)
//------------------------------------------------------------------------------
Complex& Complex::operator*=(Complex z)
{
	this->re_ = this->real() * z.real() - this->imag() * z.imag();
	this->im_ = this->real() * z.imag() + this->imag() * z.real();
}

//------------------------------------------------------------------------------
/// \details z/w = (a+bi)/(c+di) = (a+bi)(c-di)/((c+di)(c-di)) = 
/// 	((ac+bd) +i(bc-ad))/(c^2 + d^2)
//------------------------------------------------------------------------------
Complex& Complex::operator/=(Complex z)
{
	assert(modulusSquared(z) != 0.);
 
	this->re_ = (this->real() * z.real() + this->imag() * z.imag())/
		modulusSquared(z);

	this->im_ = (-this->real() * z.imag() + this->imag() * z.real())/
		modulusSquared(z);
}

//------------------------------------------------------------------------------
/// \details Many useful operations don't require direct access to 
/// 	representation of complex, so they can be defined separately from class 
/// 	definitions
//------------------------------------------------------------------------------
Complex operator+(Complex a, Complex b)
{
	return a += b;
}

Complex operator-(Complex a, Complex b)
{
	return a -= b;
}

Complex operator-(Complex a)
{
	return {-a.real(), -a.imag()};	// unary minus
}

Complex operator*(Complex a, Complex b)
{
	return a *= b;
}

Complex operator/(Complex a, Complex b)
{
	return a /= b;
}

//------------------------------------------------------------------------------
/// \details Definitions of == and != are straightforward:
//------------------------------------------------------------------------------
bool operator==(Complex a, Complex b)		// equal
{
	return a.real() == b.real() && a.imag() == b.imag();
}

bool operator!=(Complex a, Complex b) 	// not equal 
{
	return !(a==b);
}

// originally from Stroustrup, pp. 61
//Complex sqrt(Complex);
//------------------------------------------------------------------------------
// \ref https://en.wikipedia.org/wiki/Complex_number#Elementary_operations
/// \details r = |z| = |x + yi| = \sqrt{x^2 + y^2}
//------------------------------------------------------------------------------
double modulus(Complex z)
{
	return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
}

double modulusSquared(Complex z)
{
	return z.real() * z.real() + z.imag() * z.imag();
}

} // namespace Fields
