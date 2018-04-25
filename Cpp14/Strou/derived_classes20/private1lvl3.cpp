/**
 * @file   : private1lvl3.cpp
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
#include <string>

/**
 * @ref https://stackoverflow.com/questions/21886447/c-delegating-constructor-with-initialization-list-which-inits-happen
 */
B0::B0(const double a, const double b, const std::string& id,
  const double num):
//  B0{a, b, id}, num_{num} // error: mem-initializer follows constructor delegation
  x_{a}, y_{b}, id_{id}, num_{num}
{}

B0::B0(const double a, const double b, const std::string& id):
  x_{a}, y_{b}, id_{id}, num_{}
{}

const double B0::fraction() const 
{
  if (y_ != 0.)
  {
    return x_/y_;
  }
  else 
  {
    return 0.;
  }
}

const double B0::num_times_x_over_y() const
{
  return num_ * fraction();  
}
