/**
 * @file   : factorial_constexpr.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Factorial as constexpr.
 * @ref    : 12.1.6 constexpr Functions Ch. 12 Functions; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * 
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
#include <iostream> 

/**
 * @details When constexpr is used in a function definition, it means 
 * "should be usuable in a constant expression when given constant expressions as arguments." 
 * */
constexpr int fac(int n)
{
    return (n > 1) ? n * fac(n-1) : 1;
}

/** 
 * @details When used in an object definition, it means 
 * "evaluate the initializer at compile time."
 * */
void f(int n)
{
    int f5 = fac(5);    // may be evaluated at compile time 
    int fn = fac(n);    // evaluated at run time (n is a variable)

    constexpr int f6 = fac(6);  // must be evaluated at compile time 
//    constexpr int fnn = fac(n); // error: can't guarantee compile-time evaluation (n is a variable)

    char a[fac(4)];     // OK: array bounds must be constants and fac() is constexpr 
    char a2[fac(n)];    // error: array bounds must be constants and n is a variable

    // ...     
}

int main(int argc, char* argv[])
{
    constexpr int f9 = fac(9); // must be evaluated at compile time 
    constexpr int f9b { fac(9) }; // must be evaluated at compile time 
    std::cout << " f9  : " << f9 << std::endl;
    std::cout << " f9b : " << f9b << std::endl;    

    f(3);

}
