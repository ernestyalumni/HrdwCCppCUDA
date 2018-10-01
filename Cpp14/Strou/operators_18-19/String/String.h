//------------------------------------------------------------------------------
/// \file String.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate template literal operator.
/// \ref Ch. 19 Special Operators, 19.2.6 User-defined Literals
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details String provides value semantics, checked and unchecked access to
/// characters, stream I/O, support for range-for loops, equality operations,
/// and concatenation operators.
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 virtual_dtors_main.cpp -o virtual_dtors_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_STRING_H_
#define _UTILITIES_STRING_H_

#include <cstring> // std::strcpy, std::memcpy
#include <iostream>
#include <vector>

namespace Utilities
{

// \brief A simple string that implements the short string optimization
// 
// size() == sz is the number of elements
// if size()<= short_max, the characters are held in the String object itself;
// otherwise, the free store is used.
//
// ptr points to the start of the character sequence.
// The character sequence is kept zero-terminated: ptr[size()] == 0;
// This allows us to use C library string functions and to easily return a
// C-style string: c_str()
//
// To allow efficient addition of characters at end, String grows by doubling
// its allocation; capacity() is the amount of space available for characters
// (exclusing the terminating 0): sz_ + space_
//
// \details Representation for `String` chosen to meet 3 goals
// 1. To make it easy to convert a C-style string (e.g. a string literal) to a
// String and to allow easy access to the characters of a STring as a C-style
// string.
// 2. To minimize the use of the free store.
// 3. To make adding characters to the end of a String efficient.
//
// Representation supports what is known as short string optimization by using
// 2 string representations.

class String
{
  public:

    String(); // default constructor: x{""}

    explicit String(const char* p); // constructor from C-style string: x{"Euler"}

    // String has value semantics, i.e. after assignment s1 = s2, the 2 strings
    // s1 and s2 are fully distinct, and subsequent changes to 1 have no effect
    // on the other.
    String(const String&);            // copy constructor
    String& operator=(const String&); // copy assignment

    String(String&& x);               // move constructor
    String& operator=(String&& x);    // move assignment

    ~String()   // destructor
    {
      if (short_max < sz_)
      {
        delete[] ptr_;
      }
    }

    // Accessors.
    // provide const and non-const versions of the access functions to allow
    // them to be used for const as well as other objects.

    // efficient, unchecked operations with conventional [] subscript notation
    char& operator[](int n) // unchecked element access
    {
      return ptr_[n];
    }

    char operator[](int n) const // unchecked element access
    {
      return ptr_[n];
    }

    char& at(int n) // range-checked element access
    {
      check(n);
      return ptr_[n];
    }

    char at(int n) const // range-checked element access
    {
      check(n);
      return ptr_[n];
    }

    String& operator+=(char c); // add c at end

    const char* c_str() // C-style string access
    {
      return ptr_;
    }

    const char* c_str() const // C-style string access
    {
      return ptr_;
    }

    int size() const // number of elements
    {
      return sz_;
    }

    int capacity() const // elements plus available space
    {
      return (sz_ <= short_max) ? short_max : sz_ + space_;
    }

    // moves characters into newly allocated memory.
    char* expand(const char* ptr, int n) // expand into free store
    {
      char* p = new char[n];
      // char* strcpy(char* dest, const char* src)
      std::strcpy(p, ptr); // Sec. 43.4
      return p;
    }

  private:

    // If sz_ <= short_max the characters are stored in the String object
    // itself, in the array named ch.
    // If !(sz_ <= short_max), the characters are stored on the free store and
    // we may allocate extra space for expansion. The member named space is the
    // number of such characters.

    static const int short_max = 15;

    int sz_; // number of characters
    char* ptr_;

    // a form of union called an anonymous union (Sec. 8.3.2, Stroustrup),
    // which is specifically designed to allow a class to manage alternative
    // representations of objects. All members of an anonymous union are
    // allocated in the same memory, starting at the same address.
    union 
    {
      int space_;  // unused allocated space
      char ch_[short_max + 1]; // leave space for terminating 0
    };

    void check(int n) const // range check
    {
      if (n < 0 || sz_ <= n)
      {
        throw std::out_of_range("String::at()");
      }
    }

    // ancillary member functions:
    void copy_from(const String& x);
    void move_from(String& x);
};

// Ancillary Functions
// \ref Sec. 19.3.3.1 Ancillary Functions
// give String a copy of the members of another.
// Any necessary cleanup of the target String is the task of callers of
// copy_from(); copy_from() unconditionally overwrites its target.
void String::copy_from(const String& x) // make *this a copy of x
{
  if (x.sz_ <= short_max) // copy *this
  {
    // Use standard library `memcpy` (Sec. 43.5) to copy bytes of the
    // source into the target.
    std::memcpy(this, &x, sizeof(x)); // Sec. 43.5
    ptr_ = ch_;
  }
  else
  {
    ptr_ = expand(x.ptr_, x.sz_ + 1);
    sz_ = x.sz_;
    space_ = 0;
  }
}

void String::move_from(String& x)
{
  if (x.sz_ <= short_max) // copy *this
  {
    std::memcpy(this, &x, sizeof(x)); // Sec. 43.5
    ptr_ = ch_;
  }
  else // grab the elements
  {
    ptr_ = x.ptr_;
    sz_ = x.sz_;
    space_ = x.space_;
    x.ptr_ = x.ch_;  // x = ""
    x.sz_ = 0;
    x.ch_[0] = 0;
  }
}

// default constructor defines a String to be empty:
String::String(): // default constructor: x{""}
  sz_{0},
  ptr_{ch_} // ptr_ points to elements, ch_ is an initial location.
{
  ch_[0] = 0; // terminating 0
}

// ctor that takes C-style string argument must determine number of characters
// and store them appropriately.
String::String(const char* p):
  sz_{std::strlen(p)},
  ptr_{
    (sz_ <= short_max) ? ch_ : new char[sz_ + 1]},
  space_{0}
{
  std::strcpy(ptr_, p); // copy characters into ptr_ from p.
}

String::String(const String& x) // copy ctor
{
  copy_from(x);   // copy representation from x
}

String::String(String&& x) // move ctor
{
  move_from(x);
}

// copy assignment, like copy ctor, uses copy_from() to clone its argument's
// representation. In addition, it has to delete any free store owned by the
// target and make sure it doesn't get into trouble with self-assignment.
String& String::operator=(const String& x)
{
  if (this == &x) // deal with self-assignment
  {
    return *this;
  }

  char* p = (short_max < sz_) ? ptr_ : 0;
  copy_from(x);
  delete[] p;
  return *this;
}

// move assignment deletes its target's free store (if there is any and then
// moves)
String& String::operator=(String&& x)
{
  if (this == &x) // deal with self-assignment (x = move(x) is insanity)
  {
    return *this;
  }
  if (short_max < sz_)
  {
    delete[] ptr_;  // delete target
  }
  move_from(x);
  return *this;
}


namespace Strings
{

int hash(const String& s)
{
  int h {s[0]};
  for (int i {1}; i != s.size(); ++i)
  {
    h ^= s[i] >> 1; // unchecked access to s
  }
  return h;
}

void print_in_order(const String& s, const std::vector<int>& index)
{
  for (auto x : index)
  {
    std::cout << s.at(x) << '\n';
  }
}


} // namespace Strings

} // namespace Utilities

#endif // _UTILITIES_STRING_H_
