//------------------------------------------------------------------------------
/// \file member_base_initialization_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for member and base initialization examples.
/// \ref Ch. 17 Constructors; Bjarne Stroustrup, 17.4.4 In-Class Initializers
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details Initialize objects of a class with and without ctors.
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
///  g++ -std=c++14 member_base_initialization_main.cpp -o member_base_initialization_main
//------------------------------------------------------------------------------
#include <iostream>
#include <string>

/// \ref 17.4.4 In-Class Initializers, Stroustrup.

// Specify initializer for non-static data member in class declaration.
// () can't be used for in-class member initializers
class A
{
  public:
    int a {7};
    int b = 77;
};

/// above was equivalent to 
class A2
{
  public:
    int a;
    int b;
    A2():
      a{7}, b{77}
    {}
};

// often several ctors use the same initializer for a member
class A3
{
  public:
    A3():
      a{7}, b{5}, algorithm{"MD5"}, state{"Constructor run"}
    {}

    A3(int a_val):
      a{a_val}, b{5}, algorithm{"MD5"}, state{"Constructor run"}
    {}

    A3(const int a_val, const int d):
      a{7},
      b{d},
      algorithm{"MD5"},
      state{"Constructor run"}
    {}

  private:
    int a, b;
    std::string algorithm; // cryptographic hash to be applied to all As
    std::string state; // string indicating state in object life cycle
};

// to make common values explicit, we can factor out the unique initializer for data members
class A4
{
  public:
    A4():
      a{7}, b{5}
    {}

    A4(int a_val):
      a{a_val}, b{5}
    {}

    A4(const int a_val, const int d): a{7}, b{d}
    {}

  private:
    int a, b;
    std::string algorithm {"MD5"};
    std::string state {"Constructor run"};
};

// in-class member initializer can use names that are in scope at point of
// their use in member declaration. 
int count = 0;
int count2 = 0;

int f(int i)
{
  return i + count;
}

// Member initialization is done in declaration order (Sec. 17.2.3), so 
// first `m1` is initialized to value of a global variable `count2`. 

struct S
{
  int m1 {count2}; // that is, ::count2
  int m2 {f(m1)}; // that is, this->m1+::count, that is, ::count2 + ::count
  S()
  {
    ++count2; // very odd ctor
  }
};



int main()
{
  A a;
  A2 a2;
  A3 a3;
  A4 a4;

  S s1;
  ++count;
  std::cout << s1.m1 << s1.m2 << '\n'; // 00

  S s2;
  std::cout << s2.m1 << s2.m2 << '\n'; // 12

}
