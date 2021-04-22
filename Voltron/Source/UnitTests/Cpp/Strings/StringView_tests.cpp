//------------------------------------------------------------------------------
/// \file StringView_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Tests of string_view, class template describing object that can
///   refer to a constant contiguous sequence of char-like objects with first
///   element of sequence at position 0.
/// \ref https://en.cppreference.com/w/cpp/string/basic_string_view
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <boost/utility/string_view.hpp>
#include <iostream>
#include <string>
#include <string_view>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Strings)
BOOST_AUTO_TEST_SUITE(StringView_tests)

BOOST_AUTO_TEST_SUITE(BoostStringViewProperties)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromRvalueStdString)
{

}

BOOST_AUTO_TEST_SUITE_END() // BoostStringViewProperties

// cf.
// https://www.learncpp.com/cpp-tutorial/6-6a-an-introduction-to-stdstring_view/

BOOST_AUTO_TEST_SUITE(StringViewIntroduction)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsFromStringAndAnotherStringView)
{
  // Demonstrate that strings may make extra copies.
  // cf. https://www.learncpp.com/cpp-tutorial/6-6a-an-introduction-to-stdstring_view/
  {
    // Internally, this copies string "hello" 3 times, resulting in 4 copies.
    // (Maybe, there might be small string optimization for small strings, but
    // this is not always the case).
    // 1st, string literal "hello", known at compile-time and stored in the
    // binary.
    // 1 copy created when we create char[].
    // Following 2 std::string objects create 1 copy of string each.
    // Because std::string designed to be modifiable, std::string must contain
    // its own copy of string; holds true for const std::string, even though
    // they can't be modified.
    char text[] {"hello"};

    std::string str {text};
    std::string more {str};

    BOOST_TEST(std::string{text} == "hello");
    BOOST_TEST(str == "hello");
    BOOST_TEST(more == "hello");
  }

  // Unlike std::string, which keeps its own copy of the string,
  // std::string_view provides a view of a string that's defined elsewhere, i.e.
  // Object that can refer to a constant contiguous sequence of char-like
  // objects.
  {
    // View the text "hello", which is stored in the binary.
    std::string_view text {"hello"};
    std::string_view str {text}; // view of the same "hello".
    std::string_view more {str}; // view of the same "hello".

    // Output is the same, but no more copies of string "hello" are created.
    // When we copy std::string_view, the new std::string_view observes the
    // same string as copied-from std::string_view is observing.
    // std::string_view is not only fast, but has many of the functions that we
    // know from std::string.

    BOOST_TEST(text == "hello");
    BOOST_TEST(str == "hello");
    BOOST_TEST(more == "hello");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasManyFunctionsOfStdString)
{
  std::string_view str {"Trains are fast!"};

  BOOST_TEST(str.length() == 16);
  BOOST_TEST(str.substr(0, str.find(' ')) == "Trains");
  BOOST_TEST(str == "Trains are fast!");
  BOOST_TEST(!str.starts_with("Boats"));
  BOOST_TEST(str.ends_with("fast!"));
}

// Because std::string_view doesn't create a copy of the string, but rather
// share the string with string it views, if we change viewed string, changes
// are reflected in std::string_view.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IfStringChangesStringViewAppearsToChange)
{
  {
    char arr[] {"Gold"};
    std::string_view str {arr};
    BOOST_TEST(str == "Gold");

    // Change 'd' to 'f' in arr.
    arr[3] = 'f';

    BOOST_TEST(str == "Golf");
  }

  // Works on const std::string_view.
  {
    char arr[] {"Gold"};
    const std::string_view str {arr};
    BOOST_TEST(str == "Gold");

    // Change 'd' to 'f' in arr.
    arr[3] = 'f';

    BOOST_TEST(str == "Golf");
  }
}

// Best practices from
// https://www.learncpp.com/cpp-tutorial/6-6a-an-introduction-to-stdstring_view/
// Use std::string_view instead of C-style strings.
// Prefer std::string_view over std::string for read-only strings, unless you
// already have a std::string.

// std::string_view contains functions that let us manipulate the view of the
// string. This allows us to change the view without modifying the viewed
// string.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HasViewModificationFunctionsThatCannotRevertBack)
{
  std::string_view str {"Peach"};

  BOOST_TEST(str == "Peach");

  // Ignore the first characters.
  str.remove_prefix(1);

  BOOST_TEST(str == "each");

  // Ignore the last 2 characters.
  str.remove_suffix(2);

  BOOST_TEST(str == "ea");
}

BOOST_AUTO_TEST_SUITE_END() // StringViewIntroduction

// https://www.modernescpp.com/index.php/c-17-avoid-copying-with-std-string-view

BOOST_AUTO_TEST_SUITE(SmallStringOptimization)

// C++ string is like a thin wrapper that stores its data on the heap.
// Therefore, it often happens that memory allocation kicks in when you deal
// with C and C++ strings.

// Overload global operator, so to see which operation causes memory allocation.

// error: may not be declared within a namespace
/*
void* operator new(std::size_t count)
{
  std::cout << "   " << count << " bytes" << std::endl;

  // https://en.cppreference.com/w/cpp/memory/c/malloc
  // On success, returns ptr to beginning of newly allocated memory. To avoid a
  // memory leak, returned pointer must be deallocated with std::free, or
  // std::realloc().
  // On failure, return null ptr.
  return malloc(count);
}
*/

void get_string(const std::string& str)
{}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StdStringMayDoMemoryAllocation)
{
  /*
  std::cout << std::endl;

  std::cout << "std::string" << std::endl;

  */
  std::string small {"0123456789"};
  std::string substr {small.substr(5)};
  BOOST_TEST(substr == "56789");
  /*
  std::cout << "   " << substr << std::endl;

  std::cout << std::endl;

  std::cout << "get_string" << std::endl;

  get_string(small);
  get_string("0123456789");
  const char message [] = "0123456789";
  get_string(message);

  std::cout << std::endl;
  */  
}

BOOST_AUTO_TEST_SUITE_END() // SmallStringOptimization

BOOST_AUTO_TEST_SUITE_END() // StringView_tests
BOOST_AUTO_TEST_SUITE_END() // Strings
BOOST_AUTO_TEST_SUITE_END() // Cpp
