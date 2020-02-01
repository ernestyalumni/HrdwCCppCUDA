//------------------------------------------------------------------------------
/// \file PtrsArraysRefs_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Driver main file for class template String, a more general type
///   string that relies on fact that a character can be copied.
/// \details The basic mechanisms for defining and using class templates are
///   introduced through the example of a string template.
/// \ref Ch. 7 Pointers, Arrays, and References; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 7  
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 PtrsArraysRefs_main.cpp -o PtrsArraysRefs_main
//------------------------------------------------------------------------------
#include <cassert>
#include <iostream>
#include <map>
#include <memory> // std::addressof
#include <string>
#include <sstream>

template <typename T>
void oct_dec_hex_print(T x)
{
  std::cout << " x : " << x << '\n';
  std::cout << " in octal : " << std::oct << x << '\n';
  std::cout << " in decimal : " << std::dec << x << '\n';
  std::cout << " in hex : " << std::hex << x << '\n';
}

int main()
{
  {
    char c {'a'};

    char* p {&c}; // p holds the address of c; & is address-of operator

    oct_dec_hex_print(c);
    std::cout << " &c : " << std::hex << &c << '\n';
    // https://www.viva64.com/en/k/0019/
    printf("%p\n", &c);
    printf("%x\n", &c);
    printf("%p\n", (void *) &c);
    printf("%x\n", (void *) &c);

    oct_dec_hex_print(&c);

    printf("p : %p\n", p);
    printf("&p : %p\n", &p);
    printf("p : %p\n", (void *) p);
    printf("&p : %p\n", (void *) &p);

    std::cout << " addressof(p) : " << std::addressof(p) << std::hex << '\n';

    // https://stackoverflow.com/questions/22250067/how-to-get-address-of-a-pointer-in-c-c
    char** pp {&p};
    printf("pp : %p\n", pp);

    char c2 = *p; // c2 == 'a'
    std::cout << (c2 == 'a') << '\n';
    assert(c2 == 'a');


    // cf. https://en.cppreference.com/w/cpp/memory/addressof

    // cf. pp 172. Stroustrup;
    char** ppc; // pointer to pointer to char
    int* ap[15]; // array of 15 pointers to ints
    int (*fp)(char*); // pointer to function taking char* argument, returns int
    int* f(char*); // function taking char* argument, returns ptr to int
  }
  
  // cf. pp. 172, Stroustrup, 7.2.1 void*
  // The primary use for void* is for passing ptrs to functions that aren't
  // allowed to make assumptions about type of object, and for returning untyped
  // objects from functions.
  // To use such an object, we must use explicit type conversion.
  {
    auto f = [](int* pi)
    {
      void* pv = pi; // ok: implicit conversion of int* to void*
      int* pi2 = static_cast<int*>(pv); // explicit conversion back to int*
      return pi2;
    };

    int x {5};
    int* pi = &x;
    int* pi3 = f(pi);
    std::cout << " *pi3 : " << *pi3 << '\n';

    // pp. 173, when used for optimization, void* can be hidden behind
    // type-safe interface (27.3.1)
  }
  // 7.2.2 nullptr, ptr that doesn't point to an object
  {
    int* pi = nullptr;
    double* pd = nullptr;
  }

}
