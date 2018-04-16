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

#include <stdexcept> // std::out_of_range

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

constexpr const int* addr(const int& r) { return &r; } // OK

constexpr int low { 0 };
constexpr int high { 99 };

constexpr int check(int i)
{
    return (low <= i && i < high) ? i : throw std::out_of_range("Not in range.");
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
    int test_r { 32 };
    addr(test_r);

    /**
     * in particular, it can be tricky to determine whether 
     * result of such a function is a constant expression 
     * */
    static const int x { 5 }; 
    constexpr const int* p1 { addr(x) };    // OK
    constexpr int xx { *p1 };               // OK

    static int y;
    constexpr const int* p2 { addr(y) };    // OK
//    constexpr int yy { *y };                // error: attempt to read a variable

    static const int y2 { 32 };
//    constexpr int yy { *y2 }; // error: invalid type argument of unary '*' (have 'int')
    constexpr int yy { *p1 };
    std::cout << " yy : " << yy << std::endl;

    addr(5);
//    constexpr const int* tp { addr }; // error: cannot convert 'const int* (*)(const int&)' to 'const int* const' in initialization



}
