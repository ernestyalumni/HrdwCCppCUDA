/**
 * @file   : public2lvl2.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program 10.1: A program to illustrate inheritance
 * @ref    : pp. 245 Exploration 37 Inheritance, 
 * Ray Lischner. Exploring C++11 (Expert's Voice in C++).  2nd. ed. Apress (2013).
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
#include <string>

class C0
{
  public:
    
    explicit C0(const std::string& id, const std::string&);

    C0() = default;

    C0(const C0&) = default;
    C0(C0&&) = default;
    C0& operator=(const C0&) = default;
    C0& operator=(C0&&) = default;

    ~C0();

    /*************************************************************************/
    /** Accessors */
    /*************************************************************************/
    const std::string& id() const
    {
      return id_;
    }

    const std::string& name() const
    {
      return name_;
    } 

  private:
    std::string id_;
    std::string name_;
};

class C1 : public C0
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

class C2 : public C0
{
  public:
    explicit C2(const std::string& id, const std::string& name,
      const float x, const float y, const std::string& date);

    C2(); 

    C2(const C2&) = default;
    C2(C2&&) = default;
    C2& operator=(const C2&) = default;
    C2& operator=(C2&&) = default;

    ~C2();

  private:
    float x_;
    float y_;
    std::string date_;
};