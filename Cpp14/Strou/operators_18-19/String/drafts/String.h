//------------------------------------------------------------------------------
/// \file String.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate template literal operator.
/// \ref Ch. 19 Special Operators, 19.3 A String Class
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
///  g++ -std=c++14 String_main.cpp -o String_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_STRING_H_
#define _UTILITIES_STRING_H_

#include <cstring> // std::strcpy, std::memcpy
#include <iostream>
#include <stdexcept> // std::out_of_range
#include <vector>

namespace Utilities
{

//------------------------------------------------------------------------------
/// \brief A simple string that implements the short string optimization
/// 
/// size() == sz is the number of elements
/// if size()<= short_max, the characters are held in the String object itself;
/// otherwise, the free store is used.
///
/// ptr points to the start of the character sequence.
/// The character sequence is kept zero-terminated: ptr[size()] == 0;
/// This allows us to use C library string functions and to easily return a
/// C-style string: c_str()
///
/// To allow efficient addition of characters at end, String grows by doubling
/// its allocation; capacity() is the amount of space available for characters
/// (exclusing the terminating 0): sz_ + space_
///
/// \details Representation for `String` chosen to meet 3 goals
/// 1. To make it easy to convert a C-style string (e.g. a string literal) to a
/// String and to allow easy access to the characters of a STring as a C-style
/// string.
/// 2. To minimize the use of the free store.
/// 3. To make adding characters to the end of a String efficient.
///
/// Representation supports what is known as short string optimization by using
/// 2 string representations.
//------------------------------------------------------------------------------
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

    // EY
    String& operator=(const char*);

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

    // cannot be overloaded
    #if 0
    char* c_str() // C-style string access
    {
      return ptr_;
    }
    #endif 
    
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

    // Helper functions
    // \ref 19.3.5. Helper Functions. pp. 568
    friend std::ostream& operator<<(std::ostream& os, const String& s)
    {
      return os << s.c_str(); // Sec. 36.3.3
    }

    friend std::istream& operator>>(std::istream& is, String& s)
    {
      s = ""; // clear the target string

      // std::ws is function template that discards leading whitespace from an
      // input stream
      // \ref https://en.cppreference.com/w/cpp/io/manip/ws
      is >> std::ws; // skip whitespace (Sec. 38.4.5.1)
      char ch = '\0';
      while (is.get(ch) && !std::isspace(ch))
      {
        s += ch;
      }
      return is;
    }

    // Comparison Helper functions

    // 2 strings are equal if they are of the same size, and each element
    // is checked to be the same.
    friend bool operator==(const String& a, const String& b)
    {
      if (a.size() != b.size())
      {
        return false;
      }
      for (int i {0}; i != a.size(); ++i)
      {
        if (a[i] != b[i])
        {
          return false;
        }
      }
      return true;
    }

    friend bool operator!=(const String& a, const String& b)
    {
      return !(a==b);
    }

  private:

    //--------------------------------------------------------------------------
    // \brief moves characters into newly allocated memory.
    //--------------------------------------------------------------------------
    char* expand(const char* ptr, int n) // expand into free store
    {
      char* p = new char[n];
      // char* strcpy(char* dest, const char* src)
      std::strcpy(p, ptr); // Sec. 43.4
      return p;
    }

    // If sz_ <= short_max the characters are stored in the String object
    // itself, in the array named ch.
    // If !(sz_ <= short_max), the characters are stored on the free store and
    // we may allocate extra space for expansion. The member named space is the
    // number of such characters.

    static const int short_max = 15;

    std::size_t sz_; // number of characters
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

//------------------------------------------------------------------------------
/// \brief default constructor defines a String to be empty:
//------------------------------------------------------------------------------
String::String(): // default constructor: x{""}
  sz_{0},
  ptr_{ch_} // ptr_ points to elements, ch_ is an initial location.
{
  ch_[0] = 0; // terminating 0
}

//------------------------------------------------------------------------------
/// \brief ctor that takes C-style string argument must determine number of
/// characters and store them appropriately.
//------------------------------------------------------------------------------
String::String(const char* p):
  sz_{std::strlen(p)},
  ptr_{(sz_ <= short_max) ? ch_ : new char[sz_ + 1]},
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

//------------------------------------------------------------------------------
/// \brief copy assignment, like copy ctor, uses copy_from() to clone its
/// argument's representation. In addition, it has to delete any free store
/// owned by the target and make sure it doesn't get into trouble with
/// self-assignment.
//------------------------------------------------------------------------------
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

// EY
String& String::operator=(const char* p)
{
  *this = String{p};
  return *this;
}

//------------------------------------------------------------------------------
/// \brief move assignment deletes its target's free store (if there is any and
/// then moves)
//------------------------------------------------------------------------------
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
  move_from(x); // does not throw

  return *this;
}

// Adds a character to the end of the string, increasing its size by one.
String& String::operator+=(char c)
{
  if (sz_ == short_max) // expand to long string
  {
    int n = sz_ + sz_ + 2; // double the allocation (+2 because of the
      // terminating 0)

    ptr_ = expand(ptr_, n);
    space_ = n - sz_ - 2;
  }
  else if (short_max < sz_)
  {
    if (space_ == 0) // expand in free store
    {
      int n = sz_ + sz_ + 2; // double the allocation (+2 because of the
        // terminating 0)
      // expand() called to allocate the needed, more space, and move old
      // characters into the new space.
      char* p = expand(ptr_, n);

      // If there was an old allocation that needs deleting, it's returned, so
      // that += can delete it.
      delete[] ptr_;
      ptr_ = p;
      space_ = n - sz_ - 2;
    }
    else
    {
      --space_;
    }
    ptr_[sz_] = c; // add c at end
    ptr_[++sz_] = 0; // increase size and set terminator

    return *this;
  }
}


// To support the range-for loop, we need begin(), end() (Sec. 9.5.1)
// Provide those as freestanding (nonmember) functions without direct access to
// the String implementation:
#if 0
char* begin(String& x) // C-string style access
{
  return x.c_str();
}

char* end(String& x)
{
  return x.c_str() + x.size();
}
#endif 

const char* begin(const String& x)
{
  return x.c_str();
}

const char* end(const String& x)
{
  return x.c_str() + x.size();
}


// Given member function += that adds a character at the end, concatenation
// operators are easily provided as nonmember functions:
String& operator+=(String& a, const String& b) // concatenation
{
  for (auto x : b)
  {
    a += x;
  }
  return a;
}

String operator+(const String& a, const String& b) // concatenation
{
  String res {a};
  res += b;
  return res;
}

// Adding _s as a string literal suffix meaning String
String operator"" _s(const char* p, size_t)
{
  return String{p};
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
