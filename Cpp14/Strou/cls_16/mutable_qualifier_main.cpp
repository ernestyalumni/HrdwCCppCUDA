//------------------------------------------------------------------------------
/// \file mutable_qualifier_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Examples demonstrating mutable qualifier.
/// \ref https://en.cppreference.com/w/cpp/language/cv  
/// https://www.geeksforgeeks.org/c-mutable-keyword/
/// 16.2.9.3 mutable Ch. 16 Classes; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 16
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <string>

class A
{
  public:

    A(const std::string& a,
      const std::string& b,
      const int s,
      const int t):
      a_{a},
      b_{b},
      s_{s},
      t_{t}
    {}

    void set_b(const std::string& b) const
    {
      b_ = b;
    }

    void set_t(const int t)
    {
      t_ = t;
    }

    void print() const
    {
      std::cout << "a is: " << a_ << '\n';
      std::cout << "b is: " << b_ << '\n';
      std::cout << "s is: " << s_ << '\n';
      std::cout << "t is: " << t_ << std::endl;
    }

  private:

    std::string a_;
    mutable std::string b_;
    int s_;
    int t_;
};

/// Stroustrup, 16.2.9.3. pp. 463.
class Date
{
  public:
    Date():
      cache_valid_{false},
      cache_{"Start string"}
    {}

    std::string string_rep() const;

  private:

    mutable bool cache_valid_;
    mutable std::string cache_;
    void compute_cache_value() const
    {
      cache_ = "changed date";     // fill (mutable) cache
    }
};

std::string Date::string_rep() const
{
  if (!cache_valid_)
  {
    compute_cache_value();
    cache_valid_ = true;
  }
  return cache_;
}

/// Stroustrup, 16.2.9.4. Mutability through Indirectionpp. 463.
// \details More compilicated cases are often better handled by placing the
// changing data in a separate object and accessing it indirectly.
class Date2
{
  public:

#if 0
    Date2():
      c_{false, "Start date"}
    {}
#endif 

    std::string string_rep() const;   // string representation

    struct Cache 
    {
      bool valid;
      std::string rep;
    };

  private:

    std::unique_ptr<Cache> c_; // initialize in ctor
    void compute_cache_value() const
    {
      c_->rep = "changed date"; // fill what cache refers to.
    }
};

std::string Date2::string_rep() const
{
  if (!c_->valid)
  {
    compute_cache_value();
    c_->valid = true;
  }
  return c_->rep;
}



int main()
{
  const A test_a("A_string", "B_string", 3, 100);
  test_a.print();

  test_a.set_b("C_string");
//  test_a.set_t(150); // error: passing ‘const A’ as ‘this’ argument discards
  // qualifiers [-fpermissive]

  test_a.print();

  A test_a_2("A_string", "B_string", 3, 100);
  test_a_2.print();
  test_a_2.set_b("C_string");
  test_a_2.set_t(150);
  test_a_2.print();

  Date d;
  const Date cd;

  std::string s1 {d.string_rep()};
  std::cout << " s1 : " << s1 << '\n';
  std::string s2 {cd.string_rep()};
  std::cout << " s2 : " << s2 << '\n';

  Date2 d2;
  const Date2 cd2;

  std::string s12 {d2.string_rep()};
  std::cout << " s12 : " << s12 << '\n';
  std::string s22 {cd2.string_rep()};
  std::cout << " s22 : " << s22 << '\n';



  return 0;
}
