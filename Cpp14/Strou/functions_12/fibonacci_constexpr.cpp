/**
 * @file   : fibonacci_constexpr.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Fibonacci as constexpr.
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
 * 3 "laws" of recursion
 * 1. base case
 * 2. must change its state and move toward base case
 * 3. must call itself, recursively.
*/
/**
 * @details Fibonacci numbers (sequence) as recurrence relation: 
 * F_n = F_{n-1} + F_{n-2}
 * F_1 = 1, F_0 = 1 (base cases)
 * */
/** 
 * @name fib
 * @details since constexpr function, cannot have branching (i.e. if, elses)
 * */
constexpr int fib_recursive(int n)
{
  return (n < 2) ? 1 : (fib_recursive(n-1) + fib_recursive(n-2));
}

constexpr int ftbl[] { 1, 2, 3, 4, 5, 8, 13};

constexpr int fib(int n)
{
  return (n < sizeof(ftbl) / sizeof(*ftbl)) ? ftbl[n] : fib(n-2) + fib(n-1);
}

int main(int argc, char* argv[])
{

    std::cout << " fib_recursive(1) : " << fib_recursive(1) << std::endl;
    std::cout << " fib_recursive(2) : " << fib_recursive(2) << std::endl;
    std::cout << " fib_recursive(13) : " << fib_recursive(13) << std::endl;
    std::cout << " fib_recursive(14) : " << fib_recursive(14) << std::endl;
    std::cout << " fib(1) : " << fib(1) << std::endl;
    std::cout << " fib(2) : " << fib(2) << std::endl;
    std::cout << " fib(13) : " << fib(13) << std::endl;
    std::cout << " fib(14) : " << fib(14) << std::endl;

}

