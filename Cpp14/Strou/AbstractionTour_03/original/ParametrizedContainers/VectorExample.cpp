/**
 * @file   : VectorExample.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Vector as a container as a demonstrative example.  
 * @details Container is an object holding a collection of elements, and this
 *	demonstrates construction and destruction
 * @ref    : 3.2.1.2 A Container, 3.2.1.3. Initializing Containers 
 * 	Ch. 3 A Tour of C++: Abstraction Mechanisms. 
 * 	Bjarne Stroustrup, The C++ Programming Language, 4th Ed.
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
#include "VectorExample.h"

#include <algorithm> 	// std::copy
#include <cstddef> 	// std::size_t
#include <initializer_list>
#include <new> 	// std::bad_array_new_length
#include <stdexcept>

namespace Containers
{

template<typename T>
VectorExample<T>::VectorExample(const int s):
	sz_{static_cast<std::size_t>(s)}
{
/*	if (s < 0)
	{
		throw std::bad_array_new_length;
		throw "Bad array new length"; // WORKS
	}*/
	try
	{
		elements_ = new T[s];
	}
	catch (std::bad_array_new_length& le)
	{
		throw le.what();
	}
}

//------------------------------------------------------------------------------
/// \details I don't like the static_cast used below, but I'm sticking with how 
/// 	Stroustrup is using an int for the size. I don't like it because it 
/// 	"narrows" the size from an unsigned long int to an int
//------------------------------------------------------------------------------
template<typename T>
VectorExample<T>::VectorExample(std::initializer_list<T> l):
	elements_{new T[l.size()]},
	sz_{l.size()}
{
	std::copy(l.begin(), l.end(), &elements_[0]);
}

// copy constructor
template<typename T>
VectorExample<T>::VectorExample(const VectorExample& a):
	elements_{new T[a.size()]}, // allocate space for elements
	sz_{a.size()}
{
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]); // copy elements
	std::cout << "Copy construct a VectorExample of size " << size() << '\n';
}

// copy assignment
template<typename T>
VectorExample<T>& VectorExample<T>::operator=(const VectorExample& a)
{
	sz_ = a.size();

	T* p = new T[sz_];
	std::copy(&a.elements_[0], &a.elements_[sz_], &p[0]);

	delete[] elements_; 	// delete old elements

	elements_ = p;

//	delete[] p; // error: double free or corruption (fasttop):
	std::cout << "Copy assign a VectorExample of size " << size() << '\n';
	return *this;
}

template<typename T>
VectorExample<T>::VectorExample(VectorExample&& a):
	elements_{a.elements_},
	sz_{a.size()}
{
	a.elements_ = nullptr;
	a.sz_ = 0;

	std::cout << "Move construct a VectorExample of size " << size() << '\n';
}

template<typename T>
VectorExample<T>& VectorExample<T>::operator=(VectorExample&& a)
{
	sz_ = a.size();
	elements_ = std::move(a.elements_);

	a.elements_ = nullptr;
	a.sz_ = 0;

	std::cout << "Move assign a VectorExample of size " << size() << '\n';
}

// Accessors
template<typename T>
T& VectorExample<T>::operator[](const int i) const
{
	if (i < 0 || size() <= i)
	{
		throw std::out_of_range{"VectorExample::operator[]"};
	}
	return elements_[i];
}

template<typename T>
void VectorExample<T>::push_back(T x_in)
{
	if (sz_ == 0)
	{
		elements_ = new T[1];
		elements_[0] = x_in;
		sz_ += 1;
		return;
	}

	T* temp_elements {new T[sz_ + 1]};

	std::copy(&elements_[0], &elements_[sz_], &temp_elements[0]);

	// copy in the new element to be pushed in
	temp_elements[sz_] = x_in;

	delete[] elements_;

	// start with new elements, new size
	sz_ += 1;
	elements_ = new T[sz_];

	// copy new elements
	std::copy(&temp_elements[0], &temp_elements[sz_], &elements_[0]);

	delete[] temp_elements;
}

template<typename T>
UniqueVectorExample<T>::UniqueVectorExample(const int s):
	sz_{static_cast<std::size_t>(s)}
{
	try
	{
		elements_ = std::make_unique<T[]>(s);
	}
	catch (std::bad_array_new_length& le)
	{
		throw le.what();
	}
	catch (...)
	{
		throw "Bad array new length";
	}
}

template<typename T>
UniqueVectorExample<T>::UniqueVectorExample(std::initializer_list<T> l):
	elements_{std::make_unique<T[]>(l.size())},
	sz_{l.size()}
{
	std::copy(l.begin(), l.end(), &elements_[0]);
}

// copy constructor
template<typename T>
UniqueVectorExample<T>::UniqueVectorExample(const UniqueVectorExample& a):
	elements_{std::make_unique<T[]>(a.size())}, // allocate space for elements
	sz_{a.size()}
{
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]); // copy elements

	std::cout << "Copy Construct a UniqueVectorExample of size " << size() << '\n';
}

// copy assignment
template<typename T>
UniqueVectorExample<T>& UniqueVectorExample<T>::operator=(
	const UniqueVectorExample& a)
{
	sz_ = a.size();

	elements_.reset();

	elements_ = std::make_unique<T[]>(sz_);
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]);

	std::cout << "Copy assignment a UniqueVectorExample of size " << size() << '\n';
	return *this;
}

template<typename T>
UniqueVectorExample<T>::UniqueVectorExample(UniqueVectorExample&& a):
	elements_{std::move(a.elements_)},
	sz_{a.size()}
{
	a.sz_ = 0;

	std::cout << "Move construct a UniqueVectorExample of size " << size() << '\n';
}

template<typename T>
UniqueVectorExample<T>& UniqueVectorExample<T>::operator=(UniqueVectorExample&& a)
{
	sz_ = a.size();
	elements_ = std::move(a.elements_);

	a.sz_ = 0;

	std::cout << "Move assign a UniqueVectorExample of size " << size() << '\n';
}

template<typename T>
void UniqueVectorExample<T>::push_back(const T& x_in)
{
	if (sz_ == 0)
	{
		elements_ = std::make_unique<T[]>(1);
		elements_[0] = x_in;
		sz_ += 1;
		return;
	}

	sz_ += 1;

	std::unique_ptr<T[]> temp_elements {std::make_unique<T[]>(sz_)};

	std::copy(&elements_[0], &elements_[sz_ - 1], &temp_elements[0]);
	temp_elements[sz_ - 1] = x_in;

	elements_ = std::make_unique<T[]>(sz_);
	elements_ = std::move(temp_elements);
}

//------------------------------------------------------------------------------
/// \brief explicit instantiations
/// \ref https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
/// \details With class templates, all implementations must be in source files 
/// 	or explicit instantiations; drawback for explicit instantiations is ALL
/// 	use cases must be first explicitly instantiations
//------------------------------------------------------------------------------
template class VectorExample<double>;

template class UniqueVectorExample<double>;

} // namespace Containers
