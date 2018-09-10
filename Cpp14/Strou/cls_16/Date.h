//------------------------------------------------------------------------------
/// \file Date.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Concrete class for Date. 
/// \ref 16.3 Concrete Classes Ch. 16 Classes; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 16       
/// \details 
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
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _CHRONO_DATE_H_
#define _CHRONO_DATE_H_

#include <iostream>
#include <string>

namespace Chrono
{

enum class Month
{
  jan = 1,
  feb,
  mar,
  apr,
  may,
  jun,
  jul,
  aug,
  sep,
  oct,
  nov,
  dec
};

class Date
{
  public:   // public interface:

    class BadDate {};  // exception class

    explicit Date(int dd = {},
      Month mm = {},
      int yy = {}); // {} means "pick a default"

    // nonmodifying functions for examining the Date:
    int day() const;
    Month month() const;
    int year() const;

    std::string string_rep() const;   // string representation
    void char_rep(char s[], int max) const; // C-style string representation

    // (modifying) functions for changing the Date:
    Date& add_year(int n);  // add n years
    Date& add_month(int n); // add n months
    Date& add_day(int n);   // add n days

  private:
    bool is_valid();  // check if this Date represents a date
    int d_, m_, y_;      // representation
};

bool is_date(int d, Month m, int y);  // true for valid date
bool is_leapyear(int y);              // true if y is a leap year

const Date& default_date();   // the default date

std::ostream& operator<<(std::ostream& os, const Date& d);  // print d to os
std::istream& operator>>(std::istream& is, Date& d);  // read Date from is into
  // d

Date::Date(int dd, Month mm, int yy):
  d_{dd},
  m_{mm},
  y_{yy}
{
  if (y == 0)
  {
    y = default_date().year();
  }

  if (m == Month{})
  {
    m = default_date().month();
  }

  if (d == 0)
  {
    d = default_date().day();
  }

  if (!is_valid())
  {
    throw BadDate();
  }
}

} // namespace Chrono

bool Chrono::is_date(int d, Month m, int y)
{
  int ndays;

  switch (m)
  {
    case Month::feb:
      ndays = 28 + is_leapyear(y);
      break;
  }

  return 1 <= d && d <= ndays;
}



#endif // _CHRONO_DATE_H_
