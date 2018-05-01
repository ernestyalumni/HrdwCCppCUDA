/**
 * @file   : VectorExample.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Vector as a container, parametrized (template).  
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

#include <cstddef> 	// std::size_t
#include <initializer_list> 	// std::initializer_list
#include <iostream> // std::cout (include only for debugging/demonstration)
#include <memory> 	// std::unique_ptr
#include <stdexcept> 	// std::out_of_range

#include "Container.h"

namespace Containers
{

template<typename T>
class VectorExample 
{
	public:

		// constructor: establish invariant, acquire resources
		VectorExample(const int s);

		// constructor: establish invariant, acquire resources
		VectorExample(const std::size_t s):		
			elements_{new T[s]},
			sz_{s}
		{
//			for (int i {0}; i != s; ++i)
	//		{
		//		elements_[i] = 0.;	// initialize elements
			//}
		}

		VectorExample(std::initializer_list<T>); 	// initialize with a list

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
		T& operator[](const int i) const;

		T& operator[](const std::size_t i) const
		{
			if (size() <= i)
			{
				throw std::out_of_range{"VectorExample::operator[]"};
			}
			return elements_[i];
		}

		std::size_t size() const
		{
			return sz_;
		}

		void push_back(T);		// add element at end increasing the size by one

	private:
		T* elements_;	// elements points to an array of sz doubles
		std::size_t sz_;
};

//------------------------------------------------------------------------------
/// \brief VectorExampleContainer implements Container (interface)
//------------------------------------------------------------------------------
template<typename T>
class VectorExampleContainer : public Container<T>
{
	public:
		VectorExampleContainer(const std::size_t s):
			v_(s)
		{}

		VectorExampleContainer(std::initializer_list<T> l):
			v_{l}
		{}

		~VectorExampleContainer()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroy VectorExampleContainer class of size " <<
				size() << '\n';
		}

		T& operator[](const int i) const 
		{
			if (i < 0 || size() <= i)
			{
				throw std::out_of_range{"VectorExampleContainer::operator[]"};
			}
			return v_[i];
		}

		T& operator[](const std::size_t i) const
		{
			if (size() <= i)
			{
				throw std::out_of_range("VectorExampleContainer::operator[]");
			}	
			return v_[i];
		}

		std::size_t size() const
		{
			return v_.size();
		}

	private:
		VectorExample<T> v_;
};

template<typename T>
class UniqueVectorExample
{
	public:
		UniqueVectorExample(const int s);		// constructor: acquire resources

		UniqueVectorExample(const std::size_t s):
			elements_{std::make_unique<T[]>(s)},
			sz_{s}
		{}

		// initialize with a list
		UniqueVectorExample(std::initializer_list<T>); 	

		~UniqueVectorExample()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroyed elements_ of UniqueVectorExample class of size "
				<< sz_ << "\n";	
		}

		UniqueVectorExample(const UniqueVectorExample<T>&);				
		UniqueVectorExample& operator=(const UniqueVectorExample<T>&);

		UniqueVectorExample(UniqueVectorExample<T>&&);							// move constructor
		UniqueVectorExample& operator=(UniqueVectorExample&&);	// move assignment

		T& operator[](const int i) const
		{
			if (i < 0 || size() <= i)
			{
				throw std::out_of_range("UniqueVectorExample::operator[]");
			}
			return elements_[i];
		}

		T& operator[](const std::size_t i) const
		{
			if (size() <= i)
			{
				throw std::out_of_range("UniqueVectorExample::operator[]");
			}
			return elements_[i];
		}

		std::size_t size() const
		{
			return sz_;
		}

		void push_back(const T&);		// add element at end increasing the size by one

	private:
		std::unique_ptr<T[]> elements_;	
		std::size_t sz_;
};

//------------------------------------------------------------------------------
/// \brief UniqueVectorExampleContainer implements Container (interface)
//------------------------------------------------------------------------------
template<typename T>
class UniqueVectorExampleContainer : public Container<T>
{
	public:
		UniqueVectorExampleContainer(const std::size_t s):
			v_(s)
		{}

		UniqueVectorExampleContainer(std::initializer_list<T> l):
			v_{l}
		{}

		~UniqueVectorExampleContainer()
		{
			// only for debugging/demonstration; comment out if desired
			std::cout << "Destroy UniqueVectorExampleContainer class of size " <<
				size() << '\n';
		}

		T& operator[](const int i) const
		{
			return v_[i];
		}

		T& operator[](const std::size_t i) const
		{
			return v_[i];
		}

		std::size_t size() const
		{
			return v_.size();
		}

	private:
		UniqueVectorExample<T> v_;
};

}	// namespace Containers

#endif // _VECTOREXAMPLE_H_
