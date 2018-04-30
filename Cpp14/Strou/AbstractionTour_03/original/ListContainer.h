/**
 * @file   : ListContainer.h
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
#ifndef _LISTCONTAINER_H_
#define _LISTCONTAINER_H_

#include <initializer_list>
#include <iostream> // std::cout (include only for debugging/demonstration)
#include <list> 		// std::list

#include "Container.h"

namespace Containers
{

class ListContainer : public Container
{
	public:
		ListContainer() // empty List
		{}

		ListContainer(std::initializer_list<double> il):
			ld_{il}
		{}

		~ListContainer()
		{
			std::cout << "Destroying ListContainer class of size " << size() << '\n';
		}

		ListContainer(const ListContainer&);
		ListContainer& operator=(const ListContainer&);

		ListContainer(ListContainer&&);							// move constructor
		ListContainer& operator=(ListContainer&&);	// move assignment

		double& operator[](int i);

		int size() const
		{
			return ld_.size();
		}

	private:
		std::list<double> ld_; 	// (standard-library) list of doubles (Sec.4.4.2)
};

} // namespace Containers

#endif // _LISTCONTAINER_H_
