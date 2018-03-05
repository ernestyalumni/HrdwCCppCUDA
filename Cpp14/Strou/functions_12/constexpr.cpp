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

/** @ref http://en.cppreference.com/w/cpp/numeric/complex */
#include <complex>

int glob; 

constexpr void bad1(int a)      // error: constexpr function cannot be void
{
    glob = a;                   // error: side effect in constexpr function
}

constexpr int bad2(int a)
{
    if (a >= 0)
    {
        return a;
    }
    else
    {
        return -a;  // error: if -statement in constexpr function
    }
}

constexpr int bad3(int a)
{
    int sum = 0;                // error: local variable in constexpr function
    for (int i = 0; i < a; i +=1)   // error: loop in constexpr function
    {
        sum *= i;
    }
}

int main(int argc, char* argv[])
{
    bad1(2);
    std::cout << " glob after 2 : " << glob << std::endl; 
    bad1(5);
    std::cout << " glob after 5 : " << glob << std::endl; 
    std::cout << " bad2(3) : " << bad2(3) << std::endl;
    std::cout << " bad2(-3) : " << bad2(3) << std::endl;

    constexpr std::complex<float> z { 2.0 }; 
}
