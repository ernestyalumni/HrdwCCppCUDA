//------------------------------------------------------------------------------
/// \file StdAddressOf_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/memory/addressof
//------------------------------------------------------------------------------
#include "UnitTests/Tools/Contains.h"
#include "Utilities/ToBytes.h"

#include <boost/test/unit_test.hpp>
#include <cstdint> // Fixed width integer types.
#include <iostream>
#include <memory>
#include <sstream>
#include <string> // std::string, std::stoi

using UnitTests::Tools::Contains;
using Utilities::ToBytes;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(Memory)
BOOST_AUTO_TEST_SUITE(AddressOf_tests)

template <class T>
struct Ptr
{
  T* pad_; // add pad to show difference between 'this' and 'data'
  T* data_;
  Ptr(T* arg) :
    pad_{nullptr},
    data_{arg}
  {}

  ~Ptr()
  {
    delete data_;
  }

  template <class U>
  friend std::ostream& operator<<(std::ostream& os, const Ptr<U>& ptr)
  {
    os << "this = " << ptr << std::endl;
    
    return os; 
  }

  T** operator&()
  {
    return &data_;
  }
};

template <class T>
std::ostream& f(Ptr<T>* p, std::ostream& os)
{
  os << "Ptr   overload called with p = " << p << '\n';

  return os;
}

std::ostream& f(int** p, std::ostream& os)
{
  os << "int** overload called with p = " << p << '\n';
  return os;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CppReferenceExample)
{
  std::stringstream string_stream;

  Ptr<int> p {new int{42}};
  f(&p, string_stream); // calls int** overload

  std::cout << string_stream.str() << "\n";

  //BOOST_TEST(Contains{string_stream.str()}("int** overload called with p ="));

  // Clears the string stream.
  string_stream.str(std::string{});

  // Obtains actual address of object or function, even in presence of
  // overloaded operator&
  f(std::addressof(p), string_stream); // calls Ptr<int>* overload, (= this)

  std::cout << string_stream.str() << "\n";

  // Clears the string stream.
  string_stream.str(std::string{});

  string_stream << std::addressof(p);
  //(&p);

  const std::string address_of_p {string_stream.str()};

  // Clears the string stream.
  string_stream.str(std::string{});

  string_stream << (&p);

  const std::string ref_of_p {string_stream.str()};

  std::cout << "address_of_p : " << address_of_p << "\n";
  std::cout << "ref_of_p : " << ref_of_p << "\n";

  // Interprets signed integer value in string str.
  // out of range.
  //const auto int_address_of_p = std::stoi(address_of_p, 0, 16);
  // base = 16 for hex.
  const auto l_address_of_p = std::stol(address_of_p, 0, 16);
  BOOST_TEST(sizeof(l_address_of_p) == sizeof(uint64_t));
  const auto ll_address_of_p = std::stoll(address_of_p, 0, 16);
  BOOST_TEST(sizeof(ll_address_of_p) == sizeof(uint64_t));

  const auto l_ref_of_p = std::stol(ref_of_p, 0, 16);

  const auto ul_address_of_p = std::stoul(address_of_p, 0, 16);
  BOOST_TEST(sizeof(ul_address_of_p) == sizeof(uint64_t));
  const auto ull_address_of_p = std::stoull(address_of_p, 0, 16);
  BOOST_TEST(sizeof(ull_address_of_p) == sizeof(uint64_t));

  const auto ul_ref_of_p = std::stoul(ref_of_p, 0, 16);

  const ToBytes l_to_bytes_address_of_p {l_address_of_p};
  const ToBytes l_to_bytes_ref_of_p {l_ref_of_p};
  const ToBytes ul_to_bytes_address_of_p {ul_address_of_p};
  const ToBytes ul_to_bytes_ref_of_p {ul_ref_of_p};

  std::cout << l_to_bytes_address_of_p.decreasing_addresses_hex() << "\n";
  std::cout << ul_to_bytes_address_of_p.decreasing_addresses_hex() << "\n";

  // 8 bits
  BOOST_TEST(
    (ul_to_bytes_ref_of_p.data() - ul_to_bytes_address_of_p.data()) == 8);
}

BOOST_AUTO_TEST_SUITE_END() // AddressOf_tests
BOOST_AUTO_TEST_SUITE_END() // Memory
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp