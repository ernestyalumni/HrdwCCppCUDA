//------------------------------------------------------------------------------
/// \file Tuples_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for class template Tuple and Value Unit.
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
///  
//------------------------------------------------------------------------------
#include "Tuples.h"

#include <iostream>
#include <stdexcept>

using Utilities::Tuples::Tuple;
using Utilities::Tuples::SymbolTable;

int main()
{

  try
  {
    SymbolTable symbol_table;
    symbol_table.declare("Pi", 3.14);
    symbol_table.declare("e", 2.72);
    symbol_table.print();
    std::cout << "Pi is " << symbol_table.get("Pi") << "\n";

    if (symbol_table.is_declared("Pi"))
    {
      std::cout << "Pi is declared\n";
    }
    else
    {
      std::cout << "Pi is not declared \n";
    }

    if (symbol_table.is_declared("nd"))
    {
      std::cout << "nd is declared\n";
    }
    else
    {
      std::cout << "nd is not declared\n";
    }
    symbol_table.set("Pi", 4.14);

    std::cout << "Pi is now " << symbol_table.get("Pi") << "\n";


    // provoke errors
    // std::cout << symbol_table.get("nd") << "\n";
    // symbol_table.set("nd", 99);
    // symbol_table.declare("Pi", 99);

  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  catch (...)
  {
    std::cerr << "Exception\n";
  }

}
