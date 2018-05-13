/**
 * @file   : noexcept_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate noexcept.
 * @ref    : pp. 364 13.5.1.1 noexcept Functions Ch. 13 Exception Handling; 
 *   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 *   http://en.cppreference.com/w/cpp/language/noexcept_spec
 * @detail : specifies whether a function could throw exceptions.
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
//------------------------------------------------------------------------------
/// \details By adding noexcept specifier, we indicate code not written to cope
/// with a throw.
//------------------------------------------------------------------------------

// whether foo is declared noexcept depends on if expression 
// T() will throw any exceptions
template <class T>
  void foo() noexcept(noexcept(T())) {}

  void bar() noexcept(true) {}
  void baz() noexcept { throw 42; } // noexcept is the same as noexcept(true)

  int main()
  {
    foo<int>(); // noexcept(noexcept(int())) => noexcept(true), so this i fine

    bar(); // fine
//    baz(); // compiles, but at runtime this calls std::terminate
  }
