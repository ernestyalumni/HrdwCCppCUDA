//------------------------------------------------------------------------------
/// \file Tuples.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Source file for class template Tuple and Value Unit.
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
#include "Tuples.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace Utilities
{

namespace Tuples
{

double SymbolTable::get(const std::string& s)
{
  for (int i {0}; i < var_table_.size(); ++i)
  {
    if (var_table_[i].t_ == s)
    {
      return var_table_[i].u_;
    }
  }
  throw std::runtime_error("Variable not found: " + s);
}

void SymbolTable::set(const std::string& s, const double d)
{
  for (int i {0}; i < var_table_.size(); ++i)
  {
    if (var_table_[i].t_ == s)
    {
      var_table_[i].u_ = d;
      return;
    }
  }
  throw std::runtime_error("Variable not found: " + s);
}

bool SymbolTable::is_declared(const std::string& s)
{
  for (int i {0}; i < var_table_.size(); ++i)
  {
    if (var_table_[i].t_ == s)
    {
      return true;
    }
  }
  return false;
}

double SymbolTable::declare(const std::string& s, const double d)
{
  if (is_declared(s))
  {
    throw std::runtime_error("Variable exists already: " + s);
  }
  var_table_.push_back(Tuple<std::string, double>(s, d));
  return d;
}

void SymbolTable::print()
{
  for (int i {0}; i < var_table_.size(); ++i)
  {
    std::cout << var_table_[i].t_ << ": " << var_table_[i].u_ << '\n';
  }
}


} // namespace Tuples

} // namespace Utilities
