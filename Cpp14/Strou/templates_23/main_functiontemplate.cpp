/**
 * @file   : main_functiontemplate.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program to demonstrate function templates.
 * @ref    : pp. 684 23.5 Function Templates, Ch. 23 Templates; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * https://www.geeksforgeeks.org/templates-cpp/
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

 /**
  * @brief One function works for all data types.  This would work 
  * even for user defined types if operator '>' is overloaded
  */ 
#include "functiontemplate.h"

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << myMax<int>(3,7) << std::endl;  // Call myMax for int
    std::cout << myMax<double>(3.0,7.0) << std::endl;   // call myMax for double
    std::cout << myMax<char>('g', 'e') << std::endl; // call myMax for char

    std::cout << myMax('g', 'e') << std::endl; // call myMax for char

    /**
     * @ref https://www.geeksforgeeks.org/template-specialization-c/
     * @brief example for function template specialization; we have general 
     * template fun() for all data types except int. For int, there's a 
     * specialized version of fun().
     * */
    fun<char>('a'); // main template fun();
    fun<int>(10); // specialized template for int type
    fun<float>(10.14); // main template fun();

    /** -------------------------------------------------------------------- 
     * @brief 
     * @ref pp. 684 23.5 Function Templates, Ch. 23 Templates; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
     * */
    std::vector<float> { 3,1,4,1,9,}


    return 0;
}