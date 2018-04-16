/**
 * @file   : private2lvl2_main.cpp
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
#include "private2lvl2.h"

#include <iostream> 

int main()
{
  /***************************************************************************/
  /** private inheritance */
  /***************************************************************************/
  B0 b0(3., -2., "1");
  b0.print();
  std::cout << " --> " << b0.sum() << '\n';
//  std::cout << " b0.fraction() : " << b0.fraction() << '\n'; // protected

  D1 d1(-1., 3., "2", 64., "D1 2",5);
  d1.print();
  std::cout << " --> " << d1.sum() << " --> " << d1.value() << '\n';

  std::cout << " d1.num_times_x_over_y() : " << d1.num_times_x_over_y() << 
    '\n';
  std::cout << "  Indeed, " <<
    (d1.num_times_x_over_y() == (-1./3.)*64.) << '\n';

  D2 d2(-1.,8., "3", "D2 d2", 6);
  d2.print();
  std::cout << " --> " << d2.sum() << " --> " << d2.value() << '\n';

  std::cout << " d2.num_times_x_over_y() : " << d2.num_times_x_over_y() << 
    '\n';
  std::cout << "  Indeed, " <<
    (d2.num_times_x_over_y() == (-1./8.)*128.) << '\n';

  std::cout << std::endl;
  return 0;
}
