/**
 * @file   : SafeBoolIdiom_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate Safe bool idiom.
 * @ref    :
 * @detail : Provide boolean tests for a class but restricting it from taking 
 *  participation in unwanted expressions.
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
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 noexcept_eg.cpp -o noexcept_eg
 * */
#include <iostream>

//------------------------------------------------------------------------------
/// \details User provided boolean conversion functions can cause more harm 
///   than benefit because it allows them to participate in expressions you 
///   wouldn't ideally want them to. If a safe conversion operator is defined
///   then 2 or more objects of unrelated classes can be compared. 
///   Type safety is compromised.  
//------------------------------------------------------------------------------

struct Testable
{
  operator bool() const
  {
    return false;
  }
};

struct AnotherTestable
{
  operator bool() const
  {
    return true;
  }
};

//------------------------------------------------------------------------------
/// \brief Solution and Sample Code
/// \details Safe bool idiom allows syntactical convenience of testing using an
///   intuitive if statement but at the same time prevents unintended 
///   statements unknowingly getting code for safe bool idiom.
//------------------------------------------------------------------------------

class Testable2
{
  //private:
    bool ok_;
    typedef void (Testable2::*bool_type)() const;
    void this_type_does_not_support_comparisons() const 
    {}


  public:
    explicit Testable2(bool b = true):
      ok_{b}
    {}

    operator bool_type() const
    {
      return ok_ ? 
        &Testable2::this_type_does_not_support_comparisons : 0;
    }


};

template <typename T>
bool operator!=(const Testable2& lhs, const T&)
{
  lhs.this_type_does_not_support_comparisons();
  return false;
}

template <typename T>
bool operator==(const Testable2& lhs, const T&)
{
  lhs.this_type_does_not_support_comparisons();
  return false;
}

class AnotherTestable2 // ... // Identical to Testable.
{
//  private:

    bool ok_;
    typedef void (AnotherTestable2::*bool_type)() const;
    void this_type_does_not_support_comparisons() const 
    {}

  public:
    explicit AnotherTestable2(bool b = false):
      ok_{b}
    {}

    operator bool_type() const
    {
      return ok_ ? 
        &AnotherTestable2::this_type_does_not_support_comparisons : 0;
    }

};

template <typename T>
bool operator!=(const AnotherTestable2& lhs, const T&)
{
  lhs.this_type_does_not_support_comparisons();
  return true;
}

template <typename T>
bool operator==(const AnotherTestable2& lhs, const T&)
{
  lhs.this_type_does_not_support_comparisons();
  return true;
}

//------------------------------------------------------------------------------
/// \brief Reusable Solution
/// \details There are 2 plausible solutions: Using a base class with a virtual
///   function for the actual logic, or a base class that knows which function
///   to call on the derived class. 
///   As virtual functions come at a cost (especially if class you're 
///   augmenting with Boolean tests doesn't contain any other virtual 
///   functions). 
//------------------------------------------------------------------------------
class safe_bool_base
{
  public:
    typedef void (safe_bool_base::*bool_type)() const;
    void this_type_does_not_support_comparisons() const 
    {}

  protected:

    safe_bool_base()
    {}

    safe_bool_base(const safe_bool_base&) 
    {}

    safe_bool_base& operator=(const safe_bool_base&) 
    {
      return *this;
    }

    ~safe_bool_base()
    {}
};

// For testability without virtual function.
template <typename T=void>
class safe_bool : private safe_bool_base
{
  // private or protected inheritance is very important here as it triggers the
  // access control violation in main.
  public:
    operator bool_type() const
    {
      return (static_cast<const T*>(this))->boolean_test()
        ? &safe_bool_base::this_type_does_not_support_comparisons : 0;
    }

  protected:
    ~safe_bool() {}
};

// For testability with a virtual function.
template<>
class safe_bool<void> : private safe_bool_base 
{
  // private or protected inheritance is very important here as it triggers the 
  // access control violation in main.
  public:
    operator bool_type() const
    {
      return boolean_test()
        ? &safe_bool_base::this_type_does_not_support_comparisons : 0;
    }
  protected:
    virtual bool boolean_test() const = 0;
    virtual ~safe_bool() {}
};

template <typename T>
bool operator==(const safe_bool<T>& lhs, bool b)
{
  return b == static_cast<bool>(lhs);
}

template <typename T>
bool operator==(bool b, const safe_bool<T>& rhs)
{
  return b == static_cast<bool>(rhs);
}

template <typename T, typename U>
bool operator==(const safe_bool<T>& lhs, const safe_bool<U>& rhs)
{
  lhs.this_type_does_not_support_comparisons();
  return false;
}

template <typename T, typename U>
bool operator!=(const safe_bool<T>& lhs, const safe_bool<U>& rhs)
{
  lhs.this_type_does_not_support_comparisons();
  return false;
}

// Here's how to use safe_bool:

class Testable_with_virtual : public safe_bool<>
{
  public:
    virtual ~Testable_with_virtual()
    {}

  protected:
    virtual bool boolean_test() const
    {
      // Perform Boolean logic here
      return true;
    }
};

class Testable_without_virtual:
  public safe_bool <Testable_without_virtual> // CRTP idiom
{
  public:

    /* NOT virtual */ bool boolean_test() const
    {
      // Perform Boolean logic here
      return false;
    }
};

int main(void)
{
  Testable a;
  AnotherTestable b;
  if (a == b) 
  {
    std::cout << " a == b :" << (a == b) << '\n';
  }

  if (a < 0)
  {
    std::cout << " a < 0 : " << (a < 0) << '\n';
  }

  // The above comparisons are accidental and are not intended but the compiler
  // happily compiles them.


  //----------------------------------------------------------------------------
  /// \details Solution and Sample Code
  //----------------------------------------------------------------------------
  Testable2 t1;
  AnotherTestable2 t2;
  if (t1) // Works as expected
  {} 
  
/*  if (t2 == t1) // Fails to compile 
  {}
  
  if (t1 < 0) // Fails to compile 
  {}
*/

  // Here's how to use safe_bool:
  Testable_with_virtual t1b, t2b;
  Testable_without_virtual p1, p2;
  if (t1b)
  {}

  if (p1 == false)
  {
    std::cout << "p1 == false\n";
  }

/*  if (p1 == p2) // Does not compile, as expected
  {}

  if (t1b != t2b) // Does not compile, as expected
  {}
*/
  
  return 0;
}
