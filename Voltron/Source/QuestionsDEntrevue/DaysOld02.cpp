//------------------------------------------------------------------------------
/// \file DaysOld02.cpp
/// \ref https://youtu.be/UyY0NjtGu7g
/// \details Given your birthday and the current date, calculate your age in
/// days. Compensate for leap days. Assume that the birthday and current date
/// are correct dates (and no time travel). Simply put, if you were born 1 Jan
/// 2012 and todays date is 2 Jan 2012 you are 1 day old.
//------------------------------------------------------------------------------
#include <cassert>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

const std::vector<unsigned int> days_of_months {
  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

const std::vector<unsigned int> days_of_months_in_leap_year {
  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

//------------------------------------------------------------------------------
/// \fn is_leap_year
/// \details Your code here. Return True or False
/// Pseudo code for this algorithm is found at
/// http://enwikipedia.org/wiki/Leap_year#Algorithm
//------------------------------------------------------------------------------
bool is_leap_year(const unsigned int year)
{
  // If year is not divisible by 4, then it's a common year.
  if (year % 4 != 0)
  {
    return false;
  }
  // else if (year is not divisible by 100), then it is a leap year
  else if (year % 100 != 0)
  {
    return true;
  }
  // else if (year is not divisible by 400), then it is a common year
  else if (year % 400 != 0)
  {
    return false;
  }
  else
  {
    return true;
  }
}

unsigned int total_days_in_common_year()
{
  return std::accumulate(
    days_of_months.begin(),
    days_of_months.end(),
    static_cast<decltype(days_of_months)::value_type>(0));
}

struct DayMonthYear
{
  unsigned int day_;
  unsigned int month_;
  unsigned int year_;
};

// Handle the case where birthday and current date are in the same year.
std::optional<unsigned int> calculate_days_for_same_years(
  const DayMonthYear birthday,
  const DayMonthYear current_date)
{
  assert(birthday.year_ <= current_date.year_);

  if (birthday.year_ < current_date.year_)
  {
    return std::nullopt;
  }

  bool is_year_leap_year {is_leap_year(birthday.year_)};

  assert((birthday.month_ <= current_date.month_) &&
    (birthday.month_ < 12) &&
    (current_date.month_ < 12));

  if (birthday.month_ == current_date.month_)
  {
    assert(birthday.day_ <= current_date.day_);

    // Check if the given day for the given month makes sense.
    unsigned int days_in_month;

    // If month isn't February,
    if (birthday.month_ != 1)
    {
      days_in_month = days_of_months[birthday.month_];
    }
    else if (is_year_leap_year)
    {
      days_in_month = days_of_months[birthday.month_] + 1;
    }
    else
    {
      days_in_month = days_of_months[birthday.month_];      
    }

    assert((birthday.day_ <= days_in_month) &&
      (current_date.day_ <= days_in_month));

    return std::make_optional<unsigned int>(current_date.day_ - birthday.day_);
  }
  else
  {
    unsigned int days_left_from_birthday;

    // Calculate the "rest of the days" left in the month of the birthday.
    if (is_year_leap_year)
    {
      const unsigned int days_in_month {
        days_of_months_in_leap_year[birthday.month_]};

      days_left_from_birthday = (days_in_month - birthday.day_);
    }
    else
    {
      const unsigned int days_in_month {days_of_months[birthday.month_]};

      days_left_from_birthday = (days_in_month - birthday.day_);     
    }

    unsigned int total_days_between_months {0};

    if (birthday.month_ - current_date.month_ > 1)
    {
      if (is_year_leap_year)
      {
        total_days_between_months = std::accumulate(
          days_of_months_in_leap_year.begin() + birthday.month_ + 1,
          days_of_months_in_leap_year.begin() + current_date.month_,
          static_cast<decltype(days_of_months)::value_type>(0));
      }
      else
      {
        total_days_between_months = std::accumulate(
          days_of_months.begin() + birthday.month_ + 1,
          days_of_months.begin() + current_date.month_,
          static_cast<decltype(days_of_months)::value_type>(0));        
      }
    }

    return std::make_optional<unsigned int>(
      days_left_from_birthday + total_days_between_months + current_date.day_);
  }
}

unsigned int days_between_dates(
  const DayMonthYear birthday,
  const DayMonthYear current_date)
{
  const unsigned int start_year {birthday.year_};
  const unsigned int end_year {current_date.year_};

  assert(start_year <= end_year);

  if (start_year == end_year)
  {
    return *calculate_days_for_same_years(birthday, current_date);
  }

  const unsigned int days_left_from_birthday_in_year {
    *calculate_days_for_same_years(
      birthday,
      DayMonthYear{
        //days_of_months[days_of_months.size() - 1],
        //days_of_months.size() - 1,
        days_of_months[11],
        11,
        start_year})};

  const unsigned int days_to_current_date_in_year {
    *calculate_days_for_same_years(
      DayMonthYear{1, 0, current_date.year_}, current_date) + 1};

  if (end_year - start_year == 1)
  {
    return days_left_from_birthday_in_year + days_to_current_date_in_year;
  }

  unsigned int days_between_years {0};

  for (unsigned int year {start_year + 1}; year < current_date.year_; ++year)
  {
    if (is_leap_year(year))
    {
      days_between_years += (total_days_in_common_year() + 1);
    }
    else
    {
      days_between_years += total_days_in_common_year();
    }
  }

  return (days_left_from_birthday_in_year +
    days_between_years + days_to_current_date_in_year);
}


int main()
{
  std::cout << "\nDays old #2\n";  

  {
    std::cout << "\n is_leap_year \n";

    std::cout << is_leap_year(1979) << "\n";

    std::cout << "\n total days in common year : " <<
      total_days_in_common_year() << "\n";

    std::cout << "Base case: " << *calculate_days_for_same_years(
      DayMonthYear{1, 0, 2012}, DayMonthYear{2, 0, 2012}) << "\n";

    std::cout << "1th case: " << *calculate_days_for_same_years(
      DayMonthYear{2, 1, 1979}, DayMonthYear{4, 1, 1979}) << "\n";

    std::cout << "fail case: " << calculate_days_for_same_years(
      DayMonthYear{12, 25, 1666}, DayMonthYear{12, 25, 1667}).has_value();

    std::cout << "\n testing: " <<
      std::accumulate(
        days_of_months.begin() + 1, days_of_months.begin() + 2, 0) << "\n";

    std::cout << "\n other cases: " << 
      *calculate_days_for_same_years(
        DayMonthYear{1, 0, 2022}, DayMonthYear{1, 1, 2022}) << " " <<
      *calculate_days_for_same_years(
        DayMonthYear{2, 0, 2022}, DayMonthYear{3, 1, 2022}) << " " << 
      *calculate_days_for_same_years(
        DayMonthYear{3, 1, 2022}, DayMonthYear{5, 3, 2022}) << " " << 
      *calculate_days_for_same_years(
        DayMonthYear{5, 2, 2022}, DayMonthYear{2, 5, 2022}) << "\n";
  }

  {
    std::cout << "\n other cases: " << 
      days_between_dates(
        DayMonthYear{1, 0, 2022}, DayMonthYear{1, 1, 2023}) << " " <<
      days_between_dates(
        DayMonthYear{2, 0, 2020}, DayMonthYear{3, 1, 2021}) << " " << 
      days_between_dates(
        DayMonthYear{3, 1, 2020}, DayMonthYear{5, 3, 2022}) << " " << 
      days_between_dates(
        DayMonthYear{5, 2, 2021}, DayMonthYear{2, 5, 2024}) << "\n";
  }
}