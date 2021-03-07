//------------------------------------------------------------------------------
/// \file References_tests.cpp
/// \brief Unit tests demonstrating std::ref, std::cref, std::reference_wrapper
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://en.cppreference.com/w/cpp/utility/functional/reference_wrapper
//------------------------------------------------------------------------------
#include "Cpp/RuleOf5.h"

#include <algorithm> // std::copy
#include <boost/test/unit_test.hpp>
#include <functional> // std::reference_wrapper
#include <iostream>
#include <iterator> // std::advance, std::begin, std::end;
#include <list>
#include <numeric>
#include <sstream> // std::ostringstream;
#include <streambuf>
#include <vector>

using Cpp::RuleOf5::RuleOf5Object;
using std::advance;
using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::endl;
using std::iota;
using std::list;
using std::ostream_iterator;
using std::ostringstream;
using std::reference_wrapper;
using std::streambuf;
using std::vector;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(FunctionObjects)
BOOST_AUTO_TEST_SUITE(ReferenceWrapper_tests)

//------------------------------------------------------------------------------
/// \details 
/// template <class T>
/// class reference_wrapper;
///
/// std::reference_wrapper is a class template that wraps a reference in a
/// copyable, assignable object. It's frequently used as a mechanism to store
/// references inside standard containers (like std::vector) which can't
/// normally hold references.
///
/// Specifically, std::reference_wrapper is a CopyConstructible, CopyAssignable
/// wrapper around a reference to object or reference to function of type T.
/// Instances of std::reference_wrapper are objects (they can be copied or
/// stored in containers) but they are implicitly convertible to T&, so they can
/// be used as arguments with functions that take underlying type by reference.
///
/// Helper functions std::ref, std::cref often used to generate
/// std::reference_wrapper objects.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StoreReferencesInsideStandardContainers)
{
  list<int> l (10);

  // Fills range [first, list) with sequentially increasing values, starting
  // with value and repetitively evaluating ++value.
  // Results in -4, -3, -2, -1, 0, ...
  iota(l.begin(), l.end(), -4);

  BOOST_TEST(l.size() == 10);
  vector<reference_wrapper<int>> v {l.begin(), l.end()};

  auto list_iterator = l.begin();

  for (int i {0}; i < l.size(); ++i)
  {
    BOOST_TEST(*list_iterator == i - 4);
    BOOST_TEST(v.at(i) == i - 4);
    ++list_iterator;
  }

  // Double the values in the initial list.
  for (int& i : l)
  {
    i *= 2;
  }

  list_iterator = l.begin();

  for (int i {0}; i < l.size(); ++i)
  {
    BOOST_TEST(*list_iterator == (i - 4) * 2);

    // Accesses same container, list l, using multiple indexes.
    BOOST_TEST(v.at(i) == (i - 4) * 2);
    advance(list_iterator, 1);
  }
}


// cf. Discovering Modern C++: An Intensive Course for Scientists, Engineers,
// and Programmers (C++ In-Depth Series).
// by Peter Gottschling. Addison-Wesley Professional; 1st edition (December 17,
// 2015).

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ImplicitlyConvertIntoReferenceWrappers)
{
  vector<reference_wrapper<vector<int>>> vv;

  vector<int> v1 {2, 3, 4}, v2 {5, 6}, v3 {7, 8};

  // cf. 4.4.3 Reference Wrapper, pp. 232, Gottschling. Discovering Modern C++
  // (2015).
  // Vectors can be inserted.
  // They are implicitly converted into reference wrappers (reference_wrapper<T>
  // contains a ctor for T& that's not explicit)

  vv.push_back(v1);
  vv.push_back(v2);
  vv.push_back(v3);
  vv.push_back(v2);
  vv.push_back(v1);

  // cf. https://stackoverflow.com/questions/4191089/how-to-unit-test-function-writing-to-stdout-stdcout

  ostringstream local_oss;
  streambuf* cout_buffer_ptr {cout.rdbuf()}; // Save previous buffer.
  cout.rdbuf(local_oss.rdbuf());

  for (const auto& vr : vv)
  {
    // copies all elements in range [first, last) starting from first and
    // proceeding to last - 1, to another range beginning at d_first.
    // std::begin returns iterator to beginning of given container or array.
    //
    // ostream_iterator(ostream_type& stream, const CharT* delim)
    //
    // std::ostream_iterator writes successive objects of type T into the
    // std::basic_ostream object for which it was constructed, using operator
    // <<.
    copy(begin(vr.get()), end(vr.get()), ostream_iterator<int>(cout, ", "));
    //cout << endl;
  }
  
  // Restore std::cout
  cout.rdbuf(cout_buffer_ptr);

  // Uncomment to see the difference.
  //cout << "\nback to default cout\n";
  //cout << local_oss.str() << "\n";

  BOOST_TEST(local_oss.str() == "2, 3, 4, 5, 6, 7, 8, 5, 6, 2, 3, 4, ");
}

BOOST_AUTO_TEST_SUITE_END() // ReferenceWrapper_tests
BOOST_AUTO_TEST_SUITE_END() // FunctionObjects
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp