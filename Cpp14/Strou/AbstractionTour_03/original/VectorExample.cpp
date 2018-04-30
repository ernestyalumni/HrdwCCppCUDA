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
#include <initializer_list>

namespace Containers
{

//------------------------------------------------------------------------------
/// \details I don't like the static_cast used below, but I'm sticking with how 
/// 	Stroustrup is using an int for the size. I don't like it because it 
/// 	"narrows" the size from an unsigned long int to an int
//------------------------------------------------------------------------------
VectorExample::VectorExample(std::initializer_list<double> l):
	elements_{new double[l.size()]},
	sz_{static_cast<int>(l.size())}
{
//	for (int i {0}, auto iter {l.begin()}; iter != l.end(); i++, iter++)
/*	int i {0};
	for (auto element_in_l : l)
	{
		elements_[i] = element_in_l;
		i++;
	}*/
	std::copy(l.begin(), l.end(), &elements_[0]);
}

// copy constructor
VectorExample::VectorExample(const VectorExample& a):
	elements_{new double[a.size()]}, // allocate space for elements
	sz_{a.size()}
{
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]); // copy elements
	std::cout << "Copy construct a VectorExample of size " << size() << '\n';
}

// copy assignment
VectorExample& VectorExample::operator=(const VectorExample& a)
{
	sz_ = a.size();

	double* p = new double[sz_];
	std::copy(&a.elements_[0], &a.elements_[sz_], &p[0]);

	delete[] elements_; 	// delete old elements

	elements_ = p;

//	delete[] p; // error: double free or corruption (fasttop):
	std::cout << "Copy assign a VectorExample of size " << size() << '\n';
	return *this;
}

VectorExample::VectorExample(VectorExample&& a):
	elements_{a.elements_},
	sz_{a.size()}
{
	a.elements_ = nullptr;
	a.sz_ = 0;

	std::cout << "Move construct a VectorExample of size " << size() << '\n';
}

VectorExample& VectorExample::operator=(VectorExample&& a)
{
	sz_ = a.size();
	elements_ = std::move(a.elements_);

	a.elements_ = nullptr;
	a.sz_ = 0;

	std::cout << "Move assign a VectorExample of size " << size() << '\n';
}

void VectorExample::push_back(double x_in)
{
	if (sz_ == 0)
	{
		elements_ = new double[1];
		elements_[0] = x_in;
		sz_ += 1;
		return;
	}

	double* temp_elements {new double[sz_ + 1]};

	// copy old elements
/*	for (int i {0}; i < sz_; i++)
	{
		temp_elements[i] = elements_[i];
	}
*/
	std::copy(&elements_[0], &elements_[sz_], &temp_elements[0]);

	// copy in the new element to be pushed in
	temp_elements[sz_] = x_in;

	delete[] elements_;

	// start with new elements, new size
	sz_ += 1;
	elements_ = new double[sz_];

	// copy new elements
/*	for (int i {0}; i < sz_; i++)
	{
		elements_[i] = temp_elements[i];
	}*/
	std::copy(&temp_elements[0], &temp_elements[sz_], &elements_[0]);

	delete[] temp_elements;
}

UniqueVectorExample::UniqueVectorExample(std::initializer_list<double> l):
	elements_{std::make_unique<double[]>(l.size())},
	sz_{static_cast<int>(l.size())}
{
/*	int i {0};
	for (auto element_in_l : l)
	{
		elements_[i] = element_in_l;
		i++;
	}*/
	std::copy(l.begin(), l.end(), &elements_[0]);
}

// copy constructor
UniqueVectorExample::UniqueVectorExample(const UniqueVectorExample& a):
	elements_{std::make_unique<double[]>(a.size())}, // allocate space for elements
	sz_{a.size()}
{
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]); // copy elements

	std::cout << "Copy Construct a UniqueVectorExample of size " << size() << '\n';
}

// copy assignment
UniqueVectorExample& UniqueVectorExample::operator=(
	const UniqueVectorExample& a)
{
	sz_ = a.size();

	// this may not be needed nor safe; may be use reset instead
//	auto* p = elements_.release();
//	delete[] p;
	elements_.reset();

	elements_ = std::make_unique<double[]>(sz_);
	std::copy(&a.elements_[0], &a.elements_[sz_], &elements_[0]);

	std::cout << "Copy assignment a UniqueVectorExample of size " << size() << '\n';
	return *this;
}

UniqueVectorExample::UniqueVectorExample(UniqueVectorExample&& a):
//	elements_{a.elements_}, // this is copy construction
	elements_{std::move(a.elements_)},
	sz_{a.size()}
{
	a.sz_ = 0;

	std::cout << "Move construct a UniqueVectorExample of size " << size() << '\n';
}

UniqueVectorExample& UniqueVectorExample::operator=(UniqueVectorExample&& a)
{
	sz_ = a.size();
	elements_ = std::move(a.elements_);

	a.sz_ = 0;

	std::cout << "Move assign a UniqueVectorExample of size " << size() << '\n';
}

void UniqueVectorExample::push_back(double x_in)
{
	if (sz_ == 0)
	{
		elements_ = std::make_unique<double[]>(1);
		elements_[0] = x_in;
		sz_ += 1;
		return;
	}

	sz_ += 1;

	//----------------------------------------------------------------------------
	/// There doesn't seem to be any way to avoid a deep copy. Here is what I 
	/// 	tried (that DID NOT WORK):
	//----------------------------------------------------------------------------
//	std::unique_ptr<double[]> temp_elements {std::make_unique<double[]>(sz_)};
//	temp_elements = std::move(elements_);
//	auto old_ptr = elements_.release();
//	delete old_ptr;
//	temp_elements[sz_ - 1] = x_in;
//	elements_ = std::make_unique<double[]>(sz_);
//	elements_ = std::move(temp_elements);
//	elements_[sz_ - 2] = x_in;

	std::unique_ptr<double[]> temp_elements {std::make_unique<double[]>(sz_)};
/*	for (int i {0}; i < (sz_-1); i++)
	{
		temp_elements[i] = elements_[i];
	}*/
	std::copy(&elements_[0], &elements_[sz_ - 1], &temp_elements[0]);
	temp_elements[sz_ - 1] = x_in;

	elements_ = std::make_unique<double[]>(sz_);
	elements_ = std::move(temp_elements);
}



} // namespace Containers
