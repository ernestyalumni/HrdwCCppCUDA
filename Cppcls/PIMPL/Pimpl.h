//------------------------------------------------------------------------------
/// \file Pimpl.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  "Pointer to implementation" or "pImpl" is a C++ programming
/// technique that removes implementation details of a class from its object
/// representation by placing them in a separate class, accessed through an
/// opaque pointer
/// \details Because private data members of a class participate in its object
/// representation, affecting size and layout, and because private member
/// functions of a class participate in overload resolution (which takes place
/// before member access checking), any change to those implementation details
/// requires recompilation of all users of the class. pImpl breaks this
/// compilation dependency.
/// \url https://en.cppreference.com/w/cpp/language/pimpl
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
///
/// \details Every bit in the value representation of the returned To object is
/// equal to the corresponding bit in the object representation of from. The
/// values of padding bits in returned To object are unspecified.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 List_main.cpp -o List_main
//------------------------------------------------------------------------------
#ifndef _IDIOMS_PIMPL_H_
#define _IDIOMS_PIMPL_H_

#include <experimental/propagate_const>
#include <memory>

namespace Idioms
{

namespace Pimpl
{

namespace Examples
{

//------------------------------------------------------------------------------
/// \class Widget
/// \url https://en.cppreference.com/w/cpp/language/pimpl
/// \details Demonstrates a pImpl with const propagation, with back-reference
/// passed as a parameter, without allocator awareness, and move-enabled without
/// runtime checks.
///-----------------------------------------------------------------------------
// Interface (widget.h)
class Widget
{
  class Impl;

  std::experimental::propagate_const<std::unique_ptr<Impl>> pImpl;

  public:

    // public API that will be forwarded to the implementation
    void draw() const;

    void draw();

    // public API that implementation has to call
    bool shown() const
    {
      return true;
    }

    Widget(int);

    // defined in the implementation file, where Impl is a complete type
    ~Widget(); 

    // Note: calling draw() on moved-from object is UB
    Widget(Widget&&) = default;
    Widget(const Widget&) = delete;

    //--------------------------------------------------------------------------
    /// \ref https://cpppatterns.com/patterns/pimpl.html
    /// \details Note that we have explicitly defaulted move ctor and assignment
    /// operator so that Widget can be moved. To make Widget copyable, we must
    /// also implement the copy ctor and assignment operator.
    //--------------------------------------------------------------------------
    Widget& operator=(Widget&&); // defined in the implementation file
    Widget& operator=(const Widget&) = delete;
}; // class widget

//------------------------------------------------------------------------------
/// \class ClassicParser
/// \url http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
/// \brief Classic PIMPL. Classic C++98 example.
///-----------------------------------------------------------------------------
class ClassicParser
{
  public:

    ClassicParser(const char *params);

    ~ClassicParser();

    void parse(const char* input);

    // Other methods related to responsbility of the class
    // ...

  private:

    ClassicParser(const ClassicParser&); // noncopyable
    ClassicParser& operator=(const ClassicParser&); // noncopyable

    class Impl; // Forward declaration of the implementation class
    Impl *impl_; // PIMPL
};

//------------------------------------------------------------------------------
/// \class MeyerParser
/// \url http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
/// \brief Scott Meyers' C++11 PIMPL; noncopyable class because defining either
/// move ctor or move assignment operator disabled generation of copy ctor/
/// assignment.
///-----------------------------------------------------------------------------
class MeyerParser
{
  public:

    MeyerParser(const char *params);
    ~MeyerParser();

    MeyerParser(MeyerParser&&) noexcept; // movable and noncopyable
    MeyerParser& operator=(MeyerParser&&) noexcept; // movable and noncopyable

    void parse(const char* input);

  private:

    class Impl; // Forward declaration of the implementation class
    std::unique_ptr<Impl> impl_;
};

} // namespace Examples

} // namespace Pimpl

} // namespace Idioms

#endif // _IDIOMS_PIMPL_H_