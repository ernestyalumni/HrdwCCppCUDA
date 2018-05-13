/**
 * @file   : StackUnwinding_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate stack unwinding.
 * @ref    : pp. 364 13.5.1 Throwing Exceptions Ch. 13 Exception Handling; 
 *   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * 	 https://msdn.microsoft.com/en-us/library/hh254939.aspx
 * @detail : stack unwinding - process of passing the exception "up the stack"
 * 	from point of throw to a handler. 
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
//------------------------------------------------------------------------------
/// \details In C++ exception mechanism, control moves from throw statement to
/// 1st catch statement that can handle the thrown type. 
/// When catch statement is reached, all automatic variables in scope between 
/// throw and catch statements are destroyed in a process called 
/// stack unwinding.
/// 1. Control reaches try statement by normal sequential execution. Guarded 
/// section in try block is executed. 
/// 2. If no exception is thrown during execution of guarded section, catch 
/// clauses follow try block aren't executed. Execution continues at statement
/// after last catch clause that follows associated try block. 
/// 3. 
/// 
/// \note If you comment out catch statement, you can observe what happens when 
/// a program terminates because of an unhandled exception.
//------------------------------------------------------------------------------

#include <iostream>
#include <string>

//class MyException();
#include <stdexcept> // std::runtime_error

#include <typeinfo> // typeid


class Dummy
{
	public:
    Dummy(const std::string& s):
      MyName_{s}
    {
      PrintMsg("Created Dummy:");
    }

    Dummy(const Dummy& other):
      MyName_{other.MyName_}
    {
      PrintMsg("Copy created Dummy:");
    }

    ~Dummy()
    {
      PrintMsg("Destroyed Dummy");
    }

    void PrintMsg(const std::string& s)
    {
      std::cout << s << MyName_ << std::endl;
    }

    const std::string& MyName() const
    {
      return MyName_;
    }

    void set_name(const std::string& s)
    {
      MyName_ = s;
    }

  private:
    std::string MyName_;
};

void C(Dummy d, int i)
{
  std::cout << "Entering FunctionC" << '\n';
  d.set_name(" C");
  throw std::runtime_error("huh");

  std::cout << "Exiting FunctionC" << std::endl;
}

void B(Dummy d, int i)
{
  std::cout << "Entering FunctionB" << '\n';
  d.set_name("B");
  C(d, i + 1);

  std::cout << "Exiting FunctionB" << std::endl;
}

void A(Dummy d, int i)
{
  std::cout << "Entering FunctionA" << '\n';
  d.set_name(" A");
  // Dummy* pd = new Dummy("new Dummy"); // Not exception safe!!!
  B(d, i + 1);
  // delete pd;
  std::cout << "Exiting FunctionA" << std::endl;
}

int main()
{
  std::cout << "Entering main" << '\n';
  try
  {
    Dummy d(" M");
    A(d, 1);
  }
  catch (const std::exception& e)
  {
    std::cout << "Caught an exception of type: '" << typeid(e).name() << "'\n";
  }

  std::cout << "Exiting main." << std::endl;

  char c;
  std::cin >> c;
}








