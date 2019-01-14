//------------------------------------------------------------------------------
/// \file Tuple2.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  2-Tuple.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details
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
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#ifndef _SPACES_TUPLE2_H_
#define _SPACES_TUPLE2_H_

#include <ostream>

namespace Spaces
{



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
class Tuple2
{
  public:

    Tuple2();

    explicit Tuple2(const float x, const float y);

    // Accessors

    const float x() const
    {
      return x_;
    }

    const float y() const
    {
      return y_;
    }

    bool operator==(const Tuple2 U) const;

    bool operator!=(const Tuple2 U) const;

    // unary arithmetic
    inline Tuple2& operator+=(const Tuple2& U);

    inline Tuple2& operator-=(const Tuple2& U);

    Tuple2& operator*=(const float);
    Tuple2& operator/=(const float);

    // binary arithmetic
    friend Tuple2 operator+(const Tuple2& T, const Tuple2& U);
    friend Tuple2 operator-(const Tuple2& T, const Tuple2& U);

    friend std::ostream& operator<<(std::ostream& os, const Tuple2& T);

  protected:

    void set_x(const float x)
    {
      x_ = x;
    }

    void set_y(const float y)
    {
      y_ = y;
    }

  private:

    float x_;
    float y_;
};


//------------------------------------------------------------------------------
/// \details Analogous to Ival_box (integer value input box)
/// \ref 21.2.1. Implementation Inheritance, pp. 614, Stroustrup
//------------------------------------------------------------------------------
class Tuple2Implementation
{
  public:

    Tuple2Implementation();

    Tuple2Implementation(const float x, const float y);

    virtual ~Tuple2Implementation()
    {}

    virtual float get_x()
    {
      is_an_element_of_ = false;
      return x_;
    }

    virtual float get_y()
    {
      is_an_element_of_ = false;
      return y_;
    }

    virtual void set_x(const float x)
    {
      is_an_element_of_ = true;
      x_ = x;
    }

    virtual void set_y(const float y)
    {
      is_an_element_of_ = true;
      y_ = y;
    }

    virtual void reset_x(const float x)
    {
      is_an_element_of_ = false;
      x_ = x;
    }

    virtual void reset_y(const float y)
    {
      is_an_element_of_ = false;
      y_ = y;
    }

    virtual bool is_an_element_of() const
    {
      return is_an_element_of_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \ref pp. 616, Stroustrup
    /// \details data members of Tuple2Implementation declared protected to
    /// allow access from derived classes.
    /// A protected member is accessible from class's own members and from
    /// members of derived classes, but not to general users (Sec. 20.5)
    //--------------------------------------------------------------------------

    float x_;
    float y_;

    bool is_an_element_of_ {false}; // changed by user using set_value()
};

class AffineVector2Implementation : public Tuple2Implementation
{
  public:

    AffineVector2Implementation(const float x, const float y);

    float get_x() override; // get value from user and deposit it in x_
    float get_y() override;
    // void prompt() override
};

} // namespace Spaces

#endif // _SPACES_TUPLE2_H_
