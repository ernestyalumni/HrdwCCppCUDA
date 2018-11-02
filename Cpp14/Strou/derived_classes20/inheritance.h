//------------------------------------------------------------------------------
/// \file inheritance.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate inheritance in C++.
/// \ref Ch. 20 Derived classes, The C++ Programming Language, 4th Ed.,
///   Stroustrup.
/// https://en.cppreference.com/book/intro/inheritance
/// \details
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
#ifndef _INHERITANCE_H_
#define _INHERITANCE_H_

namespace Inheritance
{

// \details This type of inheritance is used to create a hierarchy of objects
// that represent a concept.
// \ref https://en.cppreference.com/book/intro/inheritance
class Number
{
  public:

    Number() = default;
    Number(Number const&) = default;
    virtual ~Number() = default;

    // Declarations of 2 pure virtual methods

    // \brief A pure virtual method that's used to query if an instance of
    // Number is an imaginary number
    virtual bool isImaginaryNumber() const noexcept = 0;

    // \brief A pure virtual method that's used to query if an instance of
    // Number is a real number.
    virtual bool isRealNumber() const noexcept = 0;
};

// \details All public members of class Number are visible to all modules
// within codebase that includes both classes. Likewise, protected members of
// class Number are visible only to code within class ImaginaryNumber and
// private members of class Number aren't visible to other modules in the
// program.

class ImaginaryNumber : public Number
{
  public:

    ImaginaryNumber():
      realComponent_{0.0}, imagComponent_{0.0}
    {}

    ImaginaryNumber(ImaginaryNumber const& arg):
      realComponent_{arg.realComponent_},
      imagComponent_{arg.imagComponent_}
    {}

    virtual ~ImaginaryNumber() = default;

    // \details Implementations of the 2 pure virtual methods declared in the
    // base class Number

    virtual bool isImaginaryNumberNumber() const noexcept
    {
      return true;
    }

    virtual bool isRealNumber() const noexcept
    {
      return false;
    }

  private:

    long double realComponent_;
    long double imagComponent_;
};

// \ref https://en.wikibooks.org/wiki/C%2B%2B_Programming/Classes/Abstract_Classes
// \details An abstract class is, conceptually, a class that can't be
// instantiated and is usually implemented as a class that has 1 or more pure
// virtual (abstract) functions.
// An abstract class contains at least 1 pure virtual function.
// A pure virtual function is 1 which must be overridden by any concrete (i.e.
// non-abstract) derived class. This is indicated in declaration with syntax
// " = 0 "

class AAbstract
{
  public:

    AAbstract() = default;
    AAbstract(const AAbstract&) = default;
    virtual ~AAbstract() = default;

    virtual bool is_B1() const noexcept = 0;
    virtual bool is_B2() const noexcept = 0;
};

class B1 : public AAbstract
{
  public:

    B1():
      b1_{0.0}, b2_{0.0}
    {}

    B1(const B1& b1):
      b1_{b1.b1_},
      b2_{b1.b2_}
    {}

    virtual ~B1() = default;

    bool is_B1() const noexcept // is implicitly virtual
    {
      return true;
    }

    bool is_B2() const noexcept // is implicitly virtual
    {
      return false;
    }

  private:

    double b1_;
    double b2_;
};

} // namespace Inheritance

#endif // _INHERITANCE_H_
