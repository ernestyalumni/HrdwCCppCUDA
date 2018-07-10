//------------------------------------------------------------------------------
/// \file Tuples.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Header file for class template Tuple and Value Unit.
/// \details Chapter 19, Exercise 03 of Stroustrup's Programming Principles, 
/// which said, template class Pair to hold pair of values of any type; used to 
/// implement symbol table like the one used in calculator (Sec. 7.8)
/// \ref https://github.com/bewuethr/stroustrup_ppp/blob/master/chapter19/chapter19_ex03.cpp 
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
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_TUPLES_H_
#define _UTILITIES_TUPLES_H_

#include <string>
#include <vector>

namespace Utilities
{

namespace Tuples
{

//------------------------------------------------------------------------------
template <class T, class U> 
struct Tuple 
{
  Tuple(T t, U u) :
    t_{t}, u_{u}
  {}

  T t_;
  U u_;
};

//------------------------------------------------------------------------------
class SymbolTable
{

  public:

    // return the value of the Variable named s
    double get(const std::string& s); 
    // set the Variable named s to d
    void set(const std::string& s, const double d); 

    bool is_declared(const std::string& var); // is var already in var_table?
    // add (var, val) to var_table
    double declare(const std::string& var, const double val);

    void print(); // print table

  private:

    std::vector<Tuple<std::string, double>> var_table_;
};

} // namespace Tuples

} // namespace Utilities

#endif // _UTILITIES_TUPLES_H_
