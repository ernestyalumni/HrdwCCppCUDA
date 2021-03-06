/**
 * @file   : Finally.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Clean up after an exception 
 * @ref    : pp. 358 13.3.1 Finally Ch. 13 Exception Handling; 
 *   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * @detail : Handle resource acquisition and release problems using objects of 
 *  classes with constructors and destructors. 
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
 *  g++ -std=c++17 -c factor.cpp
 * */
#include <iostream>

template<typename F>
struct Final_action
{
	Final_action(F f):
		clean{f}
	{}

	~Final_action()
	{
		clean();
	}
	F clean;
};

template<class F>
Final_action<F> finally(F f)
{
	return Final_action<F>(f);
}

/*
 * handle undisciplined resource acquisition
 * demonstrate that arbitrary actions are possible 
 */
void test()
{
	int* p = new int{7};				// probably should use a unique_ptr (Sec. 5.2.)
	int* buf = (int*)malloc(100*sizeof(int)); 	// C-style allocation

	auto act1 = finally([&]{

    /* memory allocated and pointed to by p and buf is appropriately deleted
		 * and free()d */
		delete p;
		free(buf);		// C-style deallocation
		std::cout << "Goodbye, Cruel world!\n";
	});

	int var = 0;
	std::cout << "var = " << var << '\n';

	// nested block:
	{
		var = 1;
		auto act2 = finally([&]{
			std::cout << "finally!\n"; 
			var = 7;
		});
		std::cout << "var= " << var << '\n';
	} // act2 is invoked here

	std::cout << "var = " << var << '\n';
} // act1 is invoked here

int main()
{
	test();
}
