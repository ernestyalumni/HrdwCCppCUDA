//------------------------------------------------------------------------------
/// \file virtual_dtors_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for classes with virtual destructors.
/// \ref Ch. 17 Constructors; Bjarne Stroustrup, 17.2.5 virtual destructors
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details RAII for CUDA C-style arrays
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
///  g++ -std=c++14 virtual_dtors_main.cpp -o virtual_dtors_main
//------------------------------------------------------------------------------

#include <iostream>

// virtual destructors
// deleting an object through pointer to base invokes undefined behavior unless
// the destructor in the base class is virtual.
class Base
{
  public:

    Base()
    {
      std::cout << " Base default constructed \n";
    }

    virtual ~Base()
    {
      std::cout << " virtual Base destructor \n";
    }
};

class Derived : public Base
{
  public:
  
    Derived()
    {
      std::cout << " Derived default constructed \n";
    }

    ~Derived()
    {
      std::cout << " Derived destructed \n";
    }
};

// Pure virtual destructors
// a pure virtual dtor for, for example in a base class which needs to be made
// abstract, but has no other suitable functions that could be declared pure
// virtual. Such destructor must have a definition, since all base class dtors
// are always called when derived class is destroyed.
class AbstractBase
{
  public:

    AbstractBase()
    {
      std::cout << " AbstractBase default constructed \n";
    }

    virtual ~AbstractBase() = 0;
};

AbstractBase::~AbstractBase()
{
  std::cout << " AbstractBase destructor \n";
}

class Derived2 : public AbstractBase
{
  public:

    Derived2()
    {
      std::cout << " Derived2 default constructed \n";
    }

    ~Derived2()
    {
      std::cout << " Derived2 destructed \n";
    }
};


int main()
{
  Base* b = new Derived;
  delete b; // safe

  std::cout << " \n Derived 2 time \n";
  Derived2 obj; // ok
}
