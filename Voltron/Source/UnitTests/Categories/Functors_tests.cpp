//------------------------------------------------------------------------------
// \file Functors_tests.cpp
//------------------------------------------------------------------------------

#include <algorithm> // std::move, std::transform
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <iostream>
#include <iterator> // std::back_inserter
#include <memory>
#include <optional>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Functors)
BOOST_AUTO_TEST_SUITE(Functors_tests)

BOOST_AUTO_TEST_SUITE(FunctorExamples)

// cf. https://www.fpcomplete.com/blog/2012/07/the-functor-pattern-in-c
BOOST_AUTO_TEST_SUITE(FromFPCompleteBlog)

std::unique_ptr<int> lifted_length(std::unique_ptr<std::string> s)
{
  return std::unique_ptr<int>(new int((*s).length()));
}

template <class A, class B>
std::unique_ptr<B> fmap(std::function<B(A)> f, std::unique_ptr<A> a)
{
  return std::unique_ptr<B>(new B(f(*a)));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TypeConstructionByUniquePtrs)
{
  std::unique_ptr<std::string> u_ptr =
    std::make_unique<std::string>("demo.txt");

  BOOST_TEST_REQUIRE(*u_ptr == "demo.txt");

  // cf. https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
  // Move smart pointer into the function argument.
  std::unique_ptr<int> length_ptr = lifted_length(std::move(u_ptr));

  BOOST_TEST(*length_ptr == 8);

  std::function<int(std::string)> string_length = [](std::string s) -> int
  {
    return std::strlen(s.c_str());
  };

  std::unique_ptr<std::string> u_ptr1 =
    std::make_unique<std::string>("demo1.txt");

  auto length_ptr1 = fmap(string_length, std::move(u_ptr1));

  BOOST_TEST(*length_ptr1 == 9);  
}

std::function<int(std::string)> string_length = [](std::string s) -> int
{
  return std::strlen(s.c_str());
};

std::vector<int> lifted_length(std::vector<std::string> strings)
{
  std::vector<int> lengths;
  std::transform(
    strings.begin(),
    strings.end(),
    // std::back_inserter is convenience function template that constructs a
    // std::back_insert_iterator for container c with type deduced from type of
    // argument
    // <iterator>
    // template <class Container>
    // std::back_insert_iterator<Container> back_inserter(Container& c);
    std::back_inserter(lengths),
    string_length);
  return lengths;
}

template <class A, class B>
std::vector<B> fmap(std::function<B(A)> f, std::vector<A> as)
{
  std::vector<B> bs;
  std::transform(as.begin(), as.end(), std::back_inserter(bs), f);
  return bs;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FunctorsOnStdVectorIntroductoryExample)
{
  const std::vector<std::string> strings {"a", "bb", "ccc", "dddd"};

  const auto lengths = lifted_length(strings);

  BOOST_TEST(lengths[0] == 1);
  BOOST_TEST(lengths[1] == 2);
  BOOST_TEST(lengths[2] == 3);
  BOOST_TEST(lengths[3] == 4);

  const auto lengths_1 = fmap(string_length, strings);

  BOOST_TEST(lengths_1[0] == 1);
  BOOST_TEST(lengths_1[1] == 2);
  BOOST_TEST(lengths_1[2] == 3);
  BOOST_TEST(lengths_1[3] == 4);
}

BOOST_AUTO_TEST_SUITE_END() // FromFPCompleteBlog

BOOST_AUTO_TEST_SUITE_END() // FunctorExamples

BOOST_AUTO_TEST_SUITE_END() // Functors_tests
BOOST_AUTO_TEST_SUITE_END() // Functors
BOOST_AUTO_TEST_SUITE_END() // Categories