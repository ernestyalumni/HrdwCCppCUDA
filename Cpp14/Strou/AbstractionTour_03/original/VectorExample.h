/**
 * @file   : VectorExample.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Vector as a container as a demonstrative example.  
 * @details Container is an object holding a collection of elements, and this
 *	demonstrates construction and destruction
 * @ref    : 3.2.1.2 A Container, Ch. 3 A Tour of C++: Abstraction 
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
 *  g++ -std=c++14 VectorExample.cpp VectorExample_main.cpp -o VectorExample_main
 * */
#ifndef _VECTOREXAMPLE_H_
#define _VECTOREXAMPLE_H_

#include <initializer_list> 	// std::initializer_list
#include <iostream> // std::cout (include only for debugging/demonstration)
#include <memory> 	// std::unique_ptr

#include "Container.h"

namespace Containers
{

class VectorExample 
{
	public:
		VectorExample(int s):		// constructor: acquire resources
			elements_{new double[s]},
			sz_{s}
		{
			for (int i {0}; i != s; ++i)
			{
				elements_[i] = 0.;	// initialize elements
			}
		}

		VectorExample(std::initializer_list<double>); 	// initialize with a list

		~VectorExample()
		{
			delete[] elements_;		// destructor: release resources

			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroyed elements_ of VectorExample class of size " <<
				sz_ << "\n";	
		}

		VectorExample(const VectorExample& a); 							// copy constructor
		VectorExample& operator=(const VectorExample& a); 	// copy assignment

		VectorExample(VectorExample&&);							// move constructor
		VectorExample& operator=(VectorExample&&);	// move assignment

		// Accessors
		double& operator[](int i)
		{
			return elements_[i];
		}

		int size() const
		{
			return sz_;
		}

		void push_back(double);		// add element at end increasing the size by one

	private:
		double* elements_;	// elements points to an array of sz doubles
		int sz_;
};

//------------------------------------------------------------------------------
/// \brief VectorExampleContainer implements Container (interface)
//------------------------------------------------------------------------------
class VectorExampleContainer : public Container
{
	public:
		VectorExampleContainer(int s):
			v_(s)
		{}

		VectorExampleContainer(std::initializer_list<double> l):
			v_{l}
		{}

		~VectorExampleContainer()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroy VectorExampleContainer class of size " <<
				size() << '\n';
		}

		double& operator[](int i)
		{
			return v_[i];
		}

		int size() const
		{
			return v_.size();
		}

	private:
		VectorExample v_;
};

class UniqueVectorExample
{
	public:
		UniqueVectorExample(int s):		// constructor: acquire resources
			elements_{std::make_unique<double[]>(s)},
			sz_{s}
		{
			for (int i {0}; i != s; ++i)
			{
				elements_[i] = 0.;	// initialize elements
			}
		}

		// initialize with a list
		UniqueVectorExample(std::initializer_list<double>); 	

		~UniqueVectorExample()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroyed elements_ of UniqueVectorExample class of size "
				<< sz_ << "\n";	
		}

		UniqueVectorExample(const UniqueVectorExample&);				
		UniqueVectorExample& operator=(const UniqueVectorExample&);

		UniqueVectorExample(UniqueVectorExample&&);							// move constructor
		UniqueVectorExample& operator=(UniqueVectorExample&&);	// move assignment

		double& operator[](int i)
		{
			return elements_[i];
		}

		int size() const
		{
			return sz_;
		}

		void push_back(double);		// add element at end increasing the size by one

	private:
	// elements points to an array of sz doubles
		std::unique_ptr<double[]> elements_;	
		int sz_;
};

//------------------------------------------------------------------------------
/// \brief UniqueVectorExampleContainer implements Container (interface)
//------------------------------------------------------------------------------
class UniqueVectorExampleContainer : public Container
{
	public:
		UniqueVectorExampleContainer(int s):
			v_(s)
		{}

		UniqueVectorExampleContainer(std::initializer_list<double> l):
			v_{l}
		{}

		~UniqueVectorExampleContainer()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroy UniqueVectorExampleContainer class of size " <<
				size() << '\n';
		}

		double& operator[](int i)
		{
			return v_[i];
		}

		int size() const
		{
			return v_.size();
		}

	private:
		UniqueVectorExample v_;
};

}	// namespace Containers

#endif // _VECTOREXAMPLE_H_
