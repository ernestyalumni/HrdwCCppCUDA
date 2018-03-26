/**
 * @file   : public2lvl2.cpp
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
#include "public2lvl2.h"
#include <iostream>
#include <string>

C0::C0(const std::string& id, const std::string& name):
  id_{id}, name_{name}
{
  std::cout << id_ << " C0\n";	
}

C0::~C0()
{
  std::cout << id_ << " ~C0\n";  
}

C1::C1(const std::string& id, const std::string& name,
  const std::string& desc, const int n):
    C0{id, name}, desc_{desc}, n_{n}
{
  std::cout << desc_ << " C1\n";
}

C1::C1():
  C0{}, desc_{}, n_{}
{
  std::cout << desc_ << " C1\n";
}

C1::~C1()
{
  std::cout << desc_ << " ~C1\n";
}

C2::C2(const std::string& id, const std::string& name,
  const float x, const float y, const std::string& date):
  C0{id, name}, x_{x}, y_{y}, date_{date}
{
  std::cout << date_ << " C2\n";
}

C2::C2()
{
  std::cout << date_ << " C2\n";
}

C2::~C2()
{
  std::cout << date_ << " ~C2\n";
}