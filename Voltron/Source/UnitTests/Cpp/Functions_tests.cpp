//------------------------------------------------------------------------------
// \file Functions_tests.cpp
//------------------------------------------------------------------------------

#include "UnitTests/Categories/Person.h"

#include <boost/test/unit_test.hpp>
#include <algorithm> // std::for_each
#include <iostream>
#include <vector>

using Categories::PersonT;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Functions_tests)

// cf. Functional Programming in C++. Ivan Čukić. November 2018
// ISBN 9781617293818 320 page Manning Publications
// Ch. 3, Function objects pp. 45

int answer = 42;
auto ask1() { return answer; } // Return type is int
const auto& ask2() { return answer; } // Return type is const int&

auto factorial(int n)
{
  if (n == 0)
  {
    return 1; // Deduces return type to be int
  }
  else
  {
    return factorial(n - 1) * n; // You know factorial returns an int, and
    // multiplying 2 ints returns an int, so you're OK.
  }
}

decltype (auto) ask() { return answer; };
decltype (auto) ask0b() { return (answer); }; // Returns a reference to int:
// decltype((answer)), whereas auto would deduce just int.
decltype (auto) ask0c() { return 42 + answer; }; // Returns an int:
// decltype(42 + answer)

// cf. https://en.cppreference.com/w/cpp/language/decltype
template <typename T, typename U>
auto add(T t, U u) -> decltype(t + u) // return type depends on template
// parameter, return type can be deduced since C++14
{
  return t + u;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctionConceptsExamples)
{
  // Compiler will check type of variable you pass to return statement, which is
  // int, and replace keyword auto with it.

  // Returns a value whose type is automatically deduced.
  BOOST_TEST(ask1() == answer);
  // Returns const-ref to type that's automatically deduced.
  BOOST_TEST(ask2() == answer);

  auto ask2_result = ask2();

  BOOST_TEST(ask2_result == answer);

  answer = 43;

  BOOST_TEST(ask2_result == 42);

  // In the case of functions with multiple return statements, all of them need
  // to return results of same type. If types differ, compiler will report an
  // error.
  // After return type is deduced, it can be used in rest of function.
  // This lets you write recursive functions with automatic return type deduction:

  BOOST_TEST(factorial(0) == 1);
  BOOST_TEST(factorial(1) == 1);
  BOOST_TEST(factorial(2) == 2);
  BOOST_TEST(factorial(3) == 6);

  BOOST_TEST(ask() == answer);
  BOOST_TEST(ask0b() == answer);
  BOOST_TEST(&ask0b() == &answer);
  BOOST_TEST(ask0c() == answer + 42);

  BOOST_TEST(add(4.2, 1.3) == 5.5);

  // decltype(auto) : function return type will be decltype of returned
  // expression.
  // This is useful when you're writing generic functions that forward result of
  // another function without modifying it. In this case, you don't know in
  // advance what function will be passed to you, and you can't know whether you
  // should pass its result back to caller as value or as reference.
  // If you pass reference, it might return a reference to temporary value
  // that'll produce undefined behavior.
  // If you pass as value, it might make an unnecessary copy of result.
}

// Perfect Forwarding for arguments.

template <typename Object, typename Function>
decltype(auto) call_on_object(Object&& object, Function function)
{
  return function(std::forward<Object>(object));
}

// Accept object argument by value, pass it on wrapped function.
// Problem: if wrapped function accept reference to object, because it needs to
// change that object, change won't be visible outside call_on_object_by_value
// because it'll be performed on local copy of object.
template <typename Object, typename Function>
decltype(auto) call_on_object_by_value(Object object, Function function)
{
  return function(object);
}

// Option 2: pass object by reference, Object& object
// Pro: would make changes to object visible to caller of function
// Problem: if function accepts argument as const-reference.
// Caller won't be able to invoke call_on_object on constant object or
// temporary value.

// Forwarding reference is written as a double reference on a templated type.
// fwd argument is forwarding reference to type T, in the following,
// whereas value isn't (it's a normal rvalue reference)
// Forwarding reference allows you to accept both const, non-const objects, and
// temporaries.
template <typename T>
void f1(T&& fwd, int && value)
{
  return std::forward<T>(fwd)(value);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PerfectForwardingForArguments)
{
  BOOST_TEST(call_on_object(4, factorial) == 24);
}

// Function pointers
// cf. 3.1.2. pp. 49. Functional Programming in C++. Ivan Čukić. November 2018

// Function pointer is a variable that stores address of a function that can
// later be called through that pointer.
// The (runtime) polymorphism is achieved by changing which function the pointer
// points to, thus changing the behavior when that function pointer is called.
//
// All types that can be implicitly converted to a function pointer are also
// function objects, but should be avoided.
// Could be useful when you need to interface with a C library.

int askf() { return 42; }

typedef decltype(askf)* FunctionPtr;

class ConvertibleToFunctionPtr
{
  public:

    // Casting operator can return only a pointer to a function.
    // Although it can return different functions depending on certain
    // conditions, it can't pass any data to them (without resorting to dirty
    // tricks).
    operator FunctionPtr() const
    {
      return askf;
    }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctionPointersExamples)
{
  // Pointer to the function
  auto ask_ptr = &askf;

  // You can create a function ptr (ask_ptr) that points to an ordinary function
  BOOST_TEST(ask_ptr() == 42);

  // Reference to the function
  auto& ask_ref = askf;

  // Demonstrate function reference (ask_ref) that references same function, and
  // that ou can call them as if they were functions themselves.
  BOOST_TEST(ask_ref() == 42);

  // Object that can implicitly converted to a function ptr.
  ConvertibleToFunctionPtr ask_wrapper;

  // Demonstrate that you can create an object that's convertible to a function
  // pointer, and call that object as if it were a normal function.
  BOOST_TEST(ask_wrapper() == 42);
}

// Call operator overloading

// Instead of creating types that can be implicitly converted to function
// pointers, C++ provides
// creating classes and overloading their call operators, to
// create new types that behave like functions.

// cf. https://en.cppreference.com/w/cpp/language/operators#Function_call_operator
struct Sum
{
  Sum():
    sum_{0}
  {}

  void operator()(int n)
  {
    sum_ += n;
  }

  int sum_;
};

template <typename V>
Sum sum_up(V&& array)
{
  Sum s = std::for_each(array.begin(), array.end(), Sum());
  return s;
};

bool divisible_by_3(const int x)
{
  return x % 3 == 0;
}

class DivisibleByN
{
  public:

    DivisibleByN(int N):
      N_{N}
    {}

    bool operator()(const int x) const
    {
      return x % N_ == 0;
    }

  private:

    int N_;
};

// 3.1.4. pp. 52 Creating generic function objects.

// Instead of creating class template, make call operator a template member
// function. This way, you won't need to specify the type when you
// instantiate the function object.
// Compiler will automatically deduce type of argument when invoking call
// operator.
class DivisibleBy
{
  public:

    DivisibleBy(const int N) :
      N_{N}
    {}

  template <typename T>
  bool operator()(T&& object) const
  {
    return std::forward<T>(object) % N_ == 0;
  }

private:

  int N_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. https://en.cppreference.com/w/cpp/algorithm/count
BOOST_AUTO_TEST_CASE(CallOperatorOverloadingExamples)
{
  const std::vector<double> v {1.1, 2.2, 3.3};

  const auto s = sum_up(v);

  BOOST_TEST(s.sum_ == 6);

  std::vector<int> v_int {1, 2, 3, 4, 4, 3, 7, 8, 9, 10};

  // You can now use DivisibleBy function object without explicitly stating type
  // of object you'll be calling it on.

  DivisibleBy predicate2 {2};
  DivisibleBy predicate3 {3};

  // count_if returns number of elements in range [first, last) satisfying
  // criteria.
  // c is for constant in cbegin, cend
  const auto result2 = std::count_if(v_int.cbegin(), v_int.cend(), predicate2);
  BOOST_TEST(result2 == 5);

  const auto result3 = std::count_if(v_int.cbegin(), v_int.cend(), predicate3);
  BOOST_TEST(result3 == 3);
}

// 3.2 Lambdas and closures

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// cf. http://www.cplusplus.com/reference/algorithm/copy_if/
BOOST_AUTO_TEST_CASE(LambdaExamples)
{
  std::vector<int> foo {25, 15, 5, -5, -15};
  //std::vector<int> bar (foo.size());
  std::vector<int> bar;

  // copy only positive numbers:
  auto it =
    std::copy_if(
      foo.begin(),
      foo.end(),
      std::back_inserter(bar),
      [](int i){return !(i < 0);});

  //bar.resize(std::distance(bar.begin(), it)); // shrink container to new size.

  // it is a back_insert_iterator    
  it = 6;

  BOOST_TEST(bar[0] == 25);
  BOOST_TEST(bar[1] == 15);
  BOOST_TEST(bar[2] == 5);
  BOOST_TEST(bar[3] == 6);
  BOOST_TEST(bar.size() == 4);
}

// cf. 3.2.2. Under the hood of lambdas

class CompanyT
{
  public:

    std::string team_name_for(const PersonT& employee) const
    {
      return employee.name();
    }

    int count_team_members(const std::string& team_name) const;

  private:

    std::vector<PersonT> employees_;
};

int CompanyT::count_team_members(const std::string& team_name) const
{
  return std::count_if(
    employees_.cbegin(),
    employees_.cend(),
    // You need to capture "this" because you're implicitly using it when
    // calling the team_name_ for member function, and you've captured team_name
    // becuase you need to use it for comparison.
    // As before, this function object has only 1 argument: an employee. You'll
    // return whether they belong to the specified team.
    [this, &team_name](const PersonT& employee)
    {
      return team_name_for(employee) == team_name;
    });
}

// 4.2 Currying

BOOST_AUTO_TEST_SUITE(Currying_tests)

// greater function and its curried version

// greater : (double, double) -> bool

bool greater(double first, double second)
{
  return first > second;
}

// greater_curried : double -> (double -> bool)
auto greater_curried(double first)
{
  return [first](double second)
  {
    return first > second;
  };
}

void print_person(
  const PersonT& person,
  std::ostream& out,
  PersonT::OutputFormatT format)
{
  if (format == PersonT::name_only)
  {
    out << person.name() << '\n';
  }
  else if (format == PersonT::full_name)
  {
    out << person.name() << ' ' << person.surname() << '\n';
  }
}

// Setup
std::vector<PersonT> people {
  {"David", "A", PersonT::male},
  {"Jane", "B", PersonT::female},
  {"Martha", "C", PersonT::female}
};

auto print_person_curried(const PersonT& person)
{
  return [&](std::ostream& out)
  {
    return [&](PersonT::OutputFormatT format)
    {
      print_person(person, out, format);
    };
  };
};

auto curried_add3(int a)
{
  return [a](int b)
  {
    return [a, b](int c)
    {
      return a + b + c;
    };
  };
};

auto less_than =
  [](int i)
  {
    return [i](int j)
    {
      return j < i;
    };
  };

template <typename B, typename M, typename E>
inline void write_partitions(B b, M m, E e)
{
  std::copy(b, m, std::ostream_iterator<int>(std::cout, " "));
  std::cout << "  |  ";

  std::copy(m, e, std::ostream_iterator<int>(std::cout, " "));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CurryingExamples)
{
  BOOST_TEST(!greater(2, 3));
  BOOST_TEST(!greater_curried(2)(3));

  // David A
  print_person_curried(people[0])(std::cout)(PersonT::full_name); // David A
  print_person_curried(people[1])(std::cout)(PersonT::name_only); // Jane
  print_person_curried(people[2])(std::cout)(PersonT::full_name); // Martha C

  BOOST_TEST(curried_add3(3)(2)(1) == 6);
  BOOST_TEST(curried_add3(7)(4)(2) == 13);  

  // cf. https://cukic.co/2013/08/06/curry-all-over-the-c11/

  std::vector<int> ns { 1, -2, 3, -4, 5, -6, 7, -8, 9, -10};

  // Original vector
  std::copy(ns.begin(), ns.end(),
    std::ostream_iterator<int>(std::cout, " "));
  std::cout << " - Original vector" << std::endl;

  // Partitions for numbers ranging from 0 to 9
  for (int i {0}; i < 10; i++)
  {
    auto p = std::partition(ns.begin(), ns.end(), less_than(i));

    write_partitions(
      ns.begin(),
      p,
      ns.end());

    std::cout << " - Predicate: _ < " << i << std::endl;
  }

}

BOOST_AUTO_TEST_SUITE_END() // Currying_tests 

BOOST_AUTO_TEST_SUITE_END() // Functions_tests
BOOST_AUTO_TEST_SUITE_END() // Cpp