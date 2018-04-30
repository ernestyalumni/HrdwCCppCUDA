/**
 * @file   : ListContainer.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : List as a container as a demonstrative example.  
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
#include "ListContainer.h"

//#include <algorithm> 	// std::copy
#include <algorithm> 	// std::move
#include <stdexcept> 	// std::out_of_range

namespace Containers
{

double& ListContainer::operator[](int i)
{
	for (auto& x : ld_)
	{
		if (i == 0)
		{
			return x;
		}
		--i;
	}
	throw std::out_of_range("List container");
}

// copy constructor
ListContainer::ListContainer(const ListContainer& l):
	ld_{l.ld_}
{}

// copy assignment
ListContainer& ListContainer::operator=(const ListContainer& l)
{
//	std::copy(l.ld_.begin(), l.ld_.end(), ld_.begin());
	ld_ = l.ld_;
}

// move constructor
ListContainer::ListContainer(ListContainer&& l):
	ld_{std::move(l.ld_)}
{}

// move assignment
ListContainer& ListContainer::operator=(ListContainer&& l)
{
	ld_ = std::move(l.ld_);
}


} // namespace Containers
