/**
 * @file   : FileOpen_main.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : opening, reading, writing a file as RAII 
 * @ref    : pp. 26 Ch. 2 File I/O; 
 *   Robert Love, Linux System Programming,  
 * @detail : Using RAII for files. 
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
 *  g++ -std=c++14 FileOpen.cpp FileOpen_main.cpp -o FileOpen_main
 * */
#include "FileOpen.h"

#include <fcntl.h>  // O_APPEND
#include <iostream>
#include <string>

// #include <stdio.h> // STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO

int main()
{
  FileOpen fileopen0 {"../../sample_data/NASA/Extra-vehicular_Activity__EVA__-_US_and_Russia.csv"}; 

  std::cout << " flags for ::open" << '\n';
  std::cout << " O_APPEND    : " << O_APPEND << '\n';
  std::cout << " O_ASYNC     : " << O_ASYNC << '\n';
  std::cout << " O_CLOEXEC   : " << O_CLOEXEC << '\n';
  std::cout << " O_CREAT   	 : " << O_CREAT << '\n';
  std::cout << " O_DIRECT    : " << O_DIRECT << '\n';
  std::cout << " O_DIRECTORY : " << O_DIRECTORY << '\n';
  std::cout << " O_EXCL      : " << O_EXCL << '\n';
  std::cout << " O_LARGEFILE : " << O_LARGEFILE << '\n';
  std::cout << " O_NOATIME+  : " << O_NOATIME << '\n';
  std::cout << " O_NOCTTY    : " << O_NOCTTY << '\n';
  std::cout << " O_NOFOLLOW  : " << O_NOFOLLOW << '\n';
  std::cout << " O_NONBLOCK  : " << O_NONBLOCK << '\n';
  std::cout << " O_SYNC      : " << O_SYNC << '\n';

#if 0
  std::cout << " STDIN_FILENO : " << STDIN_FILENO << '\n';
  std::cout << " STDOUT_FILENO : " << STDOUT_FILENO << '\n';
  std::cout << " STDERR_FILENO : " << STDERR_FILENO << '\n';
#endif 

  // enum class OpenAccessFlags
  std::cout << " OpenAccessFlags::append : " << 
    static_cast<int>(OpenAccessFlags::append) << 
    '\n';

  std::cout << " OpenAccessFlags::async : " << 
    static_cast<int>(OpenAccessFlags::async) << '\n';
  std::cout << " OpenAccessFlags::close_on_exec : " << 
    static_cast<int>(OpenAccessFlags::close_on_exec) << '\n';
  std::cout << " OpenAccessFlags::create : " << 
    static_cast<int>(OpenAccessFlags::create) << '\n';
  std::cout << " OpenAccessFlags::directIO : " << 
    static_cast<int>(OpenAccessFlags::directIO) << '\n';
  std::cout << " OpenAccessFlags::isdirectory : " << 
    static_cast<int>(OpenAccessFlags::isdirectory) << '\n';
  std::cout << " OpenAccessFlags::exclusive_with_create : " << 
    static_cast<int>(OpenAccessFlags::exclusive_with_create) << '\n';
  std::cout << " OpenAccessFlags::large_file : " << 
    static_cast<int>(OpenAccessFlags::large_file) << '\n';
  std::cout << " OpenAccessFlags::nonblocking : " <<
    static_cast<int>(OpenAccessFlags::nonblocking) << '\n';
  std::cout << " OpenAccessFlags::truncate_to_0 : " <<
    static_cast<int>(OpenAccessFlags::truncate_to_0) << '\n';


  // 


}
