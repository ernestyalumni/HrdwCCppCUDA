/**
 * @file   : tryblock_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate try block
 * @ref    : http://en.cppreference.com/w/cpp/language/try_catch 
 *   13.1.1 Exceptions Ch. 13 Exception Handling; Bjarne Stroustrup, The C++ 
 *   Programming Language, 4th Ed. 
 *   http://en.cppreference.com/w/cpp/language/throw 
 * @details try-block - Associates 1 or more exception handlers (catch-clauses)
 * with a compound statement.
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
 *  g++ -std=c++14 -lpthread tryblock_eg.cpp -o tryblock_eg
 * */
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <thread>
#include <vector>

int do_threadtask(std::thread& th)
{
  if (!th.joinable())
  {
    return 1;
  }
  else
  {
    throw std::runtime_error("error");
  }
}

void threadtaskmaster(std::thread& th)
{
  try
  {
    auto result {do_threadtask(th)};
    std::cout << " result : " << result << '\n';
  }
  catch (const std::exception& e)
  {
    std::cout << " a standard exception was caught, with message '"
      << e.what() << "'\n";
  }
}

void hello()
{
  std::cout << "Hello Concurrent World\n";
}

int do_task(size_t n)
{
  if (n <= 32)
  {
    return 1;
  }
  else
  {
    throw std::runtime_error("error");
  }
}

void taskmaster(size_t n)
{
  try
  {
    auto result {do_task(n)};
    std::cout << " result : " << result << '\n';
  }
  catch (const std::exception& e)
  {
    std::cout << " a standard exception was caught, with message '"
      << e.what() << "'\n";
  }
}

//------------------------------------------------------------------------------
/// \ref pp. 365 13.5.1 Throwing Exceptions Ch. 13 Exception Handling; 
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
/// \details Some of the most common exceptions carry no information; name of
///   the type is sufficient to report the error.
//------------------------------------------------------------------------------
struct Some_error
{};

int main()
{
	try
	{
		std::cout << "Throwing an integer exception...\n";
    throw 42;
	}
  catch (int i)
  {
    std::cout << " the integer exception was caught, with value: " << i << 
      '\n';
  }

  try
  {
    std::cout << "Creating a vector of size 5... \n";
    std::vector<int> v(5);
    std::cout << "Accessing the 11th element of the vector...\n";
    std::cout << v.at(10); // vector::at() throws std::out_of_range
  }
  catch (const std::exception& e)
  {
    // caught by reference to base
    std::cout << " a standard exception was caught, with message '"
              << e.what() << "'\n";
  }

  //----------------------------------------------------------------------------
  /// Error message I received:
  /// a standard exception was caught, with message 'error'
  /// terminate called without an active exception
  /// Hello Concurrent World
  /// Hello Concurrent World
  /// Aborted (core dumped)
  //----------------------------------------------------------------------------

//  std::thread th(hello);
//  threadtaskmaster(th);

  taskmaster(33);

  std::cout << "\n I can still do stuff " << '\n';

  try
  {
    throw Some_error{};
  }
  catch (Some_error& some_error)
  {
    std::cout << " Whoops, caught a Some_error" << '\n';
  }
  
}
