/**
 * @file   : VectorExample_main.cpp
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
#include "VectorExample.h"

using namespace Containers;

#include <iostream>

void fct(int n)
{
	VectorExample v(n);

	// use v and v2
	for (int i {0}; i < n; i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << "\n";

	{
		VectorExample v2(2*n);
		// use v and v2
		for (int i {0}; i < n; i++)
		{
			std::cout << v[i] << " ";
		}
		std::cout << "\n";

		for (int i {0}; i < 2*n; i++)
		{
			std::cout << v2[i] << " ";
		}
		std::cout << "\n";
	}	// v2 is destroyed here

	// use v
	std::cout << v[n-1] << "\n";

	return;
} // v is destroyed here

int main()
{

	fct(2);

	// VectorExampleConstructsWithInitializerList
	VectorExample v3 {1., 2., 3., 4., 5.};

	// PushBackOnAVectorExampleAddsNewElement
	v3.push_back(6.);

	for (int i {0}; i < v3.size(); i++)
	{
		std::cout << v3[i] << " ";
	}
	std::cout << "\n" << "\n";

	// UniqueVectorExampleConstructsWithInitializerList
	UniqueVectorExample uv3 {1., 2., 3., 4., 5.};

	// PushBackOnAUniqueVectorExampleAddsNewElement
	uv3.push_back(6.);

	for (int i {0}; i < uv3.size(); i++)
	{
		std::cout << uv3[i] << " ";
	}

	// VectorExampleContainerConstructsWithInteger
	VectorExampleContainer v_container(6);

	// VectorExampleContainerConstructsWithInitializerList
	VectorExampleContainer v_container1 {0., 2., 4., 6., 8.};

	// UniqueVectorExampleContainerConstructsWithInteger
	UniqueVectorExampleContainer uv_container(7);

	// UniqueVectorExampleContainerConstructsWithInitializerList
	UniqueVectorExampleContainer uv_container1 {0., 2., 4., 6., 8.};

	std::cout << "\n VectorExampleIsCopyConstructible\n";
	// VectorExampleIsCopyConstructible
	VectorExample v3b = v3;
	for (int i {0}; i < v3b.size(); i++)
	{
		std::cout << v3[i] << " ";
		std::cout << v3b[i] << " ";
	}
	std::cout << "\n" << "\n";

	// VectorExampleCanBeCopyAssigned
	VectorExample v4(2);
	v4 = v3b;

	for (int i {0}; i < v4.size(); i++)
	{
		std::cout << v3b[i] << " ";
		std::cout << v4[i] << " ";
	}
	std::cout << "\n" << "\n";

	// UniqueVectorExampleIsCopyConstructible
	UniqueVectorExample uv3b = uv3;
	for (int i {0}; i < uv3b.size(); i++)
	{
		std::cout << uv3[i] << " ";
		std::cout << uv3b[i] << " ";
	}
	std::cout << "\n" << "\n";

	// UniqueVectorExampleCanBeCopyAssigned
	UniqueVectorExample uv4(3);
	uv4 = uv3b;

	for (int i {0}; i < uv4.size(); i++)
	{
		std::cout << uv3b[i] << " ";
		std::cout << uv4[i] << " ";
	}
	std::cout << "\n" << "\n";

	// VectorExampleIsMoveConstructible
	VectorExample v4b = std::move(v4);
	for (int i {0}; i < v4b.size(); i++)
	{
		std::cout << v4b[i] << " ";
	}
	std::cout << "\n" << "\n";

	// VectorExampleCanBeMoveAssigned
	VectorExample v5(3);
	v5 = std::move(v4b);
	for (int i {0}; i < v5.size(); i++)
	{
		std::cout << v5[i] << " ";
	}
	std::cout << "\n" << "\n";

	// UniqueVectorExampleIsMoveConstructible
	UniqueVectorExample uv4b = std::move(uv4);
	for (int i {0}; i < uv4b.size(); i++)
	{
		std::cout << uv4b[i] << " ";
	}
	std::cout << "\n" << "\n";

	// UniqueVectorExampleCanBeMoveAssigned
	UniqueVectorExample uv5(4);
	uv5 = std::move(uv4b);

	for (int i {0}; i < uv5.size(); i++)
	{
		std::cout << uv5[i] << " ";
	}
	std::cout << "\n" << "\n";



}
