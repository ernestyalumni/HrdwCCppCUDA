/**
 * @file   : limits.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program to demonstrate numerical limits.
 * @ref    : Ch. 40 Numerics; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * http://en.cppreference.com/w/cpp/types/numeric_limits
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
#include <limits> 

#include <iostream>

int main()
{
    std::cout << "type\tlowset\thighest\n";
    std::cout << "int\t"
                << std::numeric_limits<int>::lowest() << '\t'
                << std::numeric_limits<int>::max() << '\n';
    std::cout << "float\t"
                << std::numeric_limits<float>::lowest() << '\t'
                << std::numeric_limits<float>::max() << '\n';
    std::cout << "double\t"
                << std::numeric_limits<double>::lowest() << '\t'
                << std::numeric_limits<double>::max() << '\n';
}

