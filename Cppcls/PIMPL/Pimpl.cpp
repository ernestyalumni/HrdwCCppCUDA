//------------------------------------------------------------------------------
/// \file Pimpl.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Obtain a value of type To by reinterpreting the object
/// representation of from for underlying bytes.
/// \url https://en.cppreference.com/w/cpp/language/pimpl
/// \ref https://en.cppreference.com/w/cpp/numeric/bit_cast
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
// Include all headers the implementation requires.
#include "Pimpl.h"

#include <iostream>
#include <memory>

namespace Idioms
{

namespace Pimpl
{

namespace Examples
{

/// \url https://en.cppreference.com/w/cpp/language/pimpl
// implementation (widget.cpp)
class Widget::Impl
{
  int n_; // private data

  public:

    void draw(const Widget& w) const 
    {
      // this call to public member function requires the back-reference      
      if (w.shown()) 
      {
        std::cout << "drawing a const widget " << n_ << '\n';
      }
    }

    void draw(const Widget& w)
    {
      if (w.shown())
      {
        std::cout << "drawing a non-const widget " << n_ << '\n';
      }
    }

    Impl(int n):
      n_{n}
    {}
};

void Widget::draw() const
{
  pImpl->draw(*this);
}

void Widget::draw()
{
  pImpl->draw(*this);
}

Widget::Widget(int n):
  pImpl{std::make_unique<Impl>(n)}
{}

//------------------------------------------------------------------------------
/// \ref https://cpppatterns.com/patterns/pimpl.html
/// \details We have explicitly defaulted Widget's ctor which is necessary
/// because dtor needs to be able to see complete definition of Impl (in order
/// to destroy the std::unique_ptr)
//------------------------------------------------------------------------------
Widget::~Widget() = default;

//------------------------------------------------------------------------------
/// \ref https://cpppatterns.com/patterns/pimpl.html
/// \details Note that we have explicitly defaulted move ctor and assignment
/// operator so that Widget can be moved. To make Widget copyable, we must also
/// implement the copy ctor and assignment operator.
//------------------------------------------------------------------------------
Widget& Widget::operator=(Widget&&) = default;

// The actual implementation definition:
class ClassicParser::Impl
{
  public:

    Impl(const char* params)
    {
      // Actual initialization
      // ...
      ch_ = params;
    }

    void parse(const char* input)
    {
      // Actual work
      std::cout << input << '\n';
      ch_ = input;
    }

  private:

    const char* ch_; // pointer to const char's
};

// Create an implementation object in ctor
ClassicParser::ClassicParser(const char* params):
  impl_(new Impl(params))
{}

// Delete the implement in dtor
ClassicParser::~ClassicParser()
{
  delete impl_;
}

// Forward an operation to the implementation
void ClassicParser::parse(const char* input)
{
  impl_->parse(input);
}

// The actual implementation definition
class MeyerParser::Impl
{
  public:
  
    Impl(const char* params)
    {
      ch_ = params;
    }

    void parse(const char* input)
    {
      std::cout << input << '\n';
      ch_ = input;
    }

  private:

    const char* ch_; // pointer to const char's
};

// Create an implementation object
MeyerParser::MeyerParser(const char* params):
  impl_(std::make_unique<Impl>(params))
{}

//------------------------------------------------------------------------------
/// \ref http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
/// \details Tell the compiler to generate default special members which utilize
/// the power of std::unique_ptr.
/// We can do it here because the implementation class is defined at this point;
/// thus std::unique_ptr can properly handle the implementation pointer.
///
/// In other words, the only problem we have to solve is that special members of
/// std::unique_ptr unable to handle (to delete, to be precise) incomplete
/// types. Thus they must be instantiated at the point where the implementation
/// class is defined. So we force them to be instantiated in source file, rather
/// than in the header. To do so we defined MeyerParser class special members
// in the source file.
//------------------------------------------------------------------------------
MeyerParser::~MeyerParser() = default;

MeyerParser::MeyerParser(MeyerParser&&) noexcept = default;

MeyerParser& MeyerParser::operator=(MeyerParser&&) noexcept = default;

// Forward an operation to the implementation
void MeyerParser::parse(const char* input)
{
  impl_->parse(input);
}


} // namespace Examples

} // namespace Pimpl
} // namespace Idioms