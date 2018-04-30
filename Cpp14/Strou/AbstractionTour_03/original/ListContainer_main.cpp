/**
 * @file   : ListContainer_main.cpp
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
 *  g++ -std=c++14 ListContainer.cpp ListContainer_main.cpp -o ListContainer_main
 * */
#include <algorithm> 	// std::move
#include <iostream> 

#include "ListContainer.h"

using namespace Containers;

int main()
{
	// ListContainerConstructs
	ListContainer list_container;

	std::cout << list_container.size() << '\n';

	// ListContainerConstructsWithInitializerList
	ListContainer list_container1 {1, 2, 3, 4, 5};

	std::cout << list_container1.size() << '\n';

	std::cout << list_container1[2] << '\n';
	std::cout << list_container1[4] << '\n';

	// ListContainerIsCopyConstructible
	ListContainer list_container2 = list_container1;

	std::cout << list_container2.size() << '\n';

	std::cout << list_container2[2] << '\n';
	std::cout << list_container2[4] << '\n';

	// ListContainerCanBeCopyAssigned
	ListContainer list_container2b;
	list_container2b = list_container2;

	std::cout << list_container2b.size() << '\n';

	std::cout << list_container2b[2] << '\n';
	std::cout << list_container2b[4] << '\n';

	// ListContainerIsMoveConstructible
	ListContainer list_container2c = std::move(list_container2);

	std::cout << list_container2c.size() << '\n';

	std::cout << list_container2c[2] << '\n';
	std::cout << list_container2c[4] << '\n';

	// ListContainerCanBeMoveAssigned
	ListContainer list_container2d;
	list_container2d = std::move(list_container2b);

	std::cout << list_container2d.size() << '\n';

	std::cout << list_container2d[2] << '\n';
	std::cout << list_container2d[4] << '\n';


}
