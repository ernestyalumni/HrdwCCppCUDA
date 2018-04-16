/**
 * @file   : File_ptr.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : class File_ptr that acts like a FILE* 
 * @ref    : pp. 355 13.3 Resource Management Ch. 13 Exception Handling; 
 *   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * @detail : Handle resource acquisition and release problems using objects of 
 *  classes with constructors and destructors. 
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
 *  g++ -std=c++17 -c factor.cpp
 * */
#include <stdio.h> // FILE
#include <stdexcept> // runtime_error
#include <iostream>

class File_ptr
{
  public:
    File_ptr(const char* n, const char* a): // open file n
      p{fopen(n, a)}
    {
      if (p == nullptr)
      {
        throw std::runtime_error{"File_ptr: Can't open file"};
      }
    }

    File_ptr(const std::string& n, const char* a): // open file n
      File_ptr{n.c_str(), a}
    {}

    explicit File_ptr(FILE* pp):   // assume ownership of pp
      p{pp}
    {
      if (p == nullptr)
      {
        throw std::runtime_error{"File_ptr: nullptr"};
      }
    }

    // Default constructor
    File_ptr() = delete; 

    File_ptr(const File_ptr&) = delete; // copy constructor
    File_ptr(File_ptr&&) = default;     // move constructor
    File_ptr& operator=(const File_ptr&) = delete; 
    File_ptr& operator=(File_ptr&&) = default;

    ~File_ptr()
    {
      fclose(p);
    }

    operator FILE*()
    {
      return p;
    }
  private:
    FILE* p;
};

int main()
{

}
