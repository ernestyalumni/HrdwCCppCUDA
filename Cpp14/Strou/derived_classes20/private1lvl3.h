/**
 * @file   : private1lvl3.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.1: A program to illustrate inheritance
 * @ref    : pp. 181 Sec. 10.3 Inheritance Ch. 10 The Projective Plane; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006. 
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
 * COMPILATION TIPS:
 *  g++ -std=c++17 -c factor.cpp
 * */
#include <iostream>
#include <string>

class B0
{
  public:

  	explicit B0(const double a, const double b, const std::string& id);
  	explicit B0(const double a, const double b, const std::string& id,
  	  const double num);
      
    B0() = delete; // default constructor (ctor)

    B0(const B0&) = default; // copy ctor
    B0(B0&&) = default; // move ctor
    B0& operator=(const B0&) = default; // copy assignment ctor
    B0& operator=(B0&&) = default; // move assignment ctor

    virtual ~B0();

    const double sum() const
    {
      return x_ + y_;
    }

    void print() const
    {
      std::cout << "(" << x_ << "," << y_ << ")";
    }

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const double get_x_() const
    {
      return x_;
    }

    const std::string& id() const
    {
      return id_;
    }

  protected:

    const double fraction() const; 

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const double num() const
    {
      return num_;
    }

    const double num_times_x_over_y() const;

  private:
    double x_;
    double y_;
    std::string id_;
    double num_ {};
}; // END of class B0

class D1 : private B0
{
  public:
    
    using B0::get_x_;
    using B0::num_times_x_over_y;  
    using B0::sum; // without it, error: B0::sum is inaccessible

    explicit D1(const double a, const double b, const std::string& id,
      const double num,
      const std::string& name, const int k);

    D1() = delete;
    
    D1(const D1&) = default;
    D1(D1&&) = default;
    D1& operator=(const D1&) = default;
    D1& operator=(D1&&) = default;

    ~D1() = default;

    const double value() const
    {
      return sum() * k_;
    }

    void print() const;

  private:
    int k_;
    std::string name_;
}; // END of class Child_public

class C1 : public B0
{
  public:

    explicit C1(const std::string& id, const std::string& name,
      const std::string& desc, const int n);

    C1();

    C1(const C1&) = default; // copy cotr
    C1(C1&&) = default;
    C1& operator=(const C1&) = default;
    C1& operator=(C1&&) = default;

    ~C1();

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const std::string& desc() const
    {
      return desc_;
    }

    int n() const
    {
      return n_;
    } 
  
  private:
    std::string desc_;
    int n_;
};

