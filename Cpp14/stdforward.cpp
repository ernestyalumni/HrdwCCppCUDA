//------------------------------------------------------------------------------
/// \file stdforward.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Examples demonstrating std::forward and perfect forwarding.
/// \ref https://en.cppreference.com/w/cpp/utility/forward     
/// \details When `t` is a "forwarding reference" (a function argument that's
/// an rvalue reference to a cv-unqualified function template parameter), this
/// overload forwards argument to another function with the value category it
/// had when passed to the calling function. 
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
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <string>

//------------------------------------------------------------------------------
/// \brief This example demonstrates perfect forwarding of the parameter(s) to
/// the argument of the constructor of class T. Also, perfect forwarding of 
/// parameter packs is demonsrated.
//------------------------------------------------------------------------------

struct A
{
  A(int&& n)
  {
    std::cout << "rvalue overload, n=" << n << "\n";
  }

  A(int& n)
  {
    std::cout << "lvalue overload, n=" << n << "\n";
  }
};

class B
{
  public:
    template<class T1, class T2, class T3>
    B(T1&& t1, T2&& t2, T3&& t3):
      a1_{std::forward<T1>(t1)},
      a2_{std::forward<T2>(t2)},
      a3_{std::forward<T3>(t3)}
    {}

  private:
    A a1_, a2_, a3_;
};

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/utility/forward
/// \details When u is a "forwarding reference" (a function argument that is an
/// rvalue reference to a cv-unqualified function template parameter), this
/// overload forwards the argument to another function with the value category
/// (lvalue, prvalue, rvalue, etc.) it had when passed to the calling function.
//------------------------------------------------------------------------------
template <class T, class U>
std::unique_ptr<T> make_unique1(U&& u)
{
  return std::unique_ptr<T>(new T(std::forward<U>(u)));
}

template <class T, class... U>
std::unique_ptr<T> make_unique(U&&... u)
{
  return std::unique_ptr<T>(new T(std::forward<U>(u)...));
}

//------------------------------------------------------------------------------
/// \ref http://bajamircea.github.io/coding/cpp/2016/04/07/move-forward.html
//------------------------------------------------------------------------------
struct Y
{
  Y() = default;
  Y(const Y&)
  {
    std::cout << "arg copied\n";
  }
  Y(Y&&)
  {
    std::cout << "arg moved\n";
  }
};

struct X
{
  template <typename A, typename B>
  X(A&& a, B&& b):
    // retrieve the original value category from constructor call
    // and pass on to member variables
    a_{std::forward<A>(a)},
    b_{std::forward<B>(b)}
  {}

  Y a_;
  Y b_;
};

template <typename A, typename B>
X factory(A&& a, B&& b)
{
  // retrieve the original value category from the factory call
  // and pass on to X constructor
  return X(std::forward<A>(a), std::forward<B>(b));
}


class NameValue
{
  public:
    explicit NameValue(const std::string& name, const float value):
      name_{name}, value_{value}
    {}

    void run()
    {
      std::cout << " Running : " << name_ << " " << value_ << '\n';
      value_ *= 2.;
      std::cout << " New value : " << value_ << '\n';
    }

  private:
    std::string name_;
    float value_;
};

class ValueUnit
{
  public:
    explicit ValueUnit(const double value, const std::string& name):
      value_{value}, name_{name}
    {}

    void run()
    {
      std::cout << " Running : " << name_ << " " << value_ << '\n';
      value_ += 1.;
      std::cout << " New value : " << value_ << '\n';
    }

  private:
    double value_;
    std::string name_;
};

template <class Executable>
class Executor
{
  public:

    template <typename ... Args>
    explicit Executor(Args&&... arguments):
      executable_{std::forward<Args>(arguments)...}
    {}   

    void run_it()
    {
      executable_.run();
    }

  private:

    Executable executable_;
};

int main()
{
  auto p1 = make_unique1<A>(2); // rvalue
  int i = 1;
  auto p2 = make_unique1<A>(i); // lvalue

  std::cout << "B\n";
  auto t = make_unique<B>(2, i, 3);

  Y y;
  X two {factory(y, Y())};

  // the first argument is a lvalue, eventually a_ will have the 
  // copy constructor called
  // the second argument is an rvalue, eventually b_ will have the 
  // move constructor called

  NameValue name_value {"g", 9.8};
  ValueUnit value_unit {32.0, "ft/sec2"};

  name_value.run();
  value_unit.run();

  Executor<NameValue> name_value_executor {"g", 9.8f};
  name_value_executor.run_it();
  name_value_executor.run_it();

  Executor<ValueUnit> value_unit_executor {32.0, "ft/sec^2"};
  value_unit_executor.run_it();
  value_unit_executor.run_it();

}

// prints
// arg copied
// arg moved

