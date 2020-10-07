//------------------------------------------------------------------------------
/// \file ToHexString.h
/// \author Ernest Yeung
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_TO_HEX_STRING_H
#define UTILITIES_TO_HEX_STRING_H

#include <algorithm> // std::for_each
#include <iomanip> // std::setfill
#include <limits> // numeric_limits<float>
#include <list> // list
#include <numeric> // std::iota
#include <sstream> // std::stringstream
#include <string>
#include <type_traits> // std::enable_if_t
#include <utility>

static_assert(
  std::numeric_limits<float>::is_iec559,
  "unsupported floating-point format");

namespace Utilities
{

template <typename T>
struct ToHexString
{
  T value_;

  explicit ToHexString(const T& value);

  explicit ToHexString(T&& value);

  //std::string operator()(const T& value);
  std::string operator()();

  std::ostream& as_increasing_addresses(std::ostream& out) const;
  std::ostream& as_decreasing_addresses(std::ostream& out) const;

  std::string as_increasing_addresses() const;
  std::string as_decreasing_addresses() const;

  void increasing_addresses_print() const;
  void decreasing_addresses_print() const;

  static std::ostream& insert_sequentially(
    std::ostream& out,
    const std::list<unsigned int>& list_range,
    const unsigned char* x_as_uchars);

  static std::ostringstream& insert_sequentially(
    std::ostringstream& out,
    const std::list<unsigned int>& list_range,
    const unsigned char* x_as_uchars);

  T value() const
  {
    return value_;
  }
};

template <typename T>
ToHexString<T>::ToHexString(const T& value) :
  value_{value}
{}

template <typename T>
ToHexString<T>::ToHexString(T&& value) :
  value_{std::move(value)}
{}


template <typename T>
//std::string ToHexString<T>::operator()(const T& value)
std::string ToHexString<T>::operator()()
{
  //value_ = value;

  return as_increasing_addresses();
  //out << to_hex_string(value_);

//  return out.str();
}

template <typename T>
std::ostream& ToHexString<T>::insert_sequentially(
  std::ostream& out,
  const std::list<unsigned int>& list_range,
  const unsigned char* x_as_uchars)
{
  std::for_each(
    list_range.begin(),
    list_range.end(),
    [&out, &x_as_uchars](const auto& index)
    {
      std::array<char, 1> single_buffer;

      const char* format {"%01x"};

      // Writes results to a character string buffer
      std::sprintf(
        &single_buffer[0],
        format,
        x_as_uchars[index]);

      out << single_buffer.data();
    });

  return out;
}

template <typename T>
std::ostringstream& ToHexString<T>::insert_sequentially(
  std::ostringstream& out,
  const std::list<unsigned int>& list_range,
  const unsigned char* x_as_uchars)
{
  std::for_each(
    list_range.begin(),
    list_range.end(),
    [&out, &x_as_uchars](const auto& index)
    {
      std::array<char, 1> single_buffer;

      const char* format {"%01x"};

      // Writes results to a character string buffer
      std::sprintf(
        &single_buffer[0],
        format,
        x_as_uchars[index]);

      out << single_buffer.data();
    });

  return out;
}

template <typename T>
std::ostream& ToHexString<T>::as_increasing_addresses(std::ostream& out) const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  const auto x_as_uchars = reinterpret_cast<const unsigned char*>(&value_);

  // Initializer list doesn't work to make std::list have size of N.
  std::list<unsigned int> list_range (number_of_bytes);
  // https://en.cppreference.com/w/cpp/algorithm/iota
  // template <class ForwardIt, class T>
  // void iota(ForwardIt first, ForwardIt last, T value);
  // Fills the range [first, last) with sequentially increasing values, starting
  // with value and repetitively evaluating ++value.
  std::iota(list_range.begin(), list_range.end(), 0);

  insert_sequentially(out, list_range, x_as_uchars);

  return out;
}

/*
template <typename T>
std::string ToHexString<T>::as_increasing_addresses() const
{
  std::ostringstream out;

  constexpr std::size_t number_of_bytes {sizeof(T)};

  const auto x_as_uchars = reinterpret_cast<const unsigned char*>(&value_);

  std::list<unsigned int> list_range (number_of_bytes);
  std::iota(list_range.begin(), list_range.end(), 0);

  insert_sequentially(out, list_range, x_as_uchars);

  return out.str();
}
*/

template <typename T>
std::ostream& ToHexString<T>::as_decreasing_addresses(std::ostream& out) const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  const auto x_as_uchars = reinterpret_cast<const unsigned char*>(&value_);

  // Initializer list doesn't work to make std::list have size of N.
  std::list<unsigned int> list_range (number_of_bytes);
  std::generate(
    list_range.begin(),
    list_range.end(),
    [n = (number_of_bytes - 1)]() mutable
    {
      return n--;
    });

  insert_sequentially(out, list_range, x_as_uchars);

  return out;
}


template <typename T>
std::string ToHexString<T>::as_increasing_addresses() const
{
  std::stringstream string_stream;

  as_increasing_addresses(string_stream);

  return string_stream.str();
}


template <typename T>
std::string ToHexString<T>::as_decreasing_addresses() const
{
  std::stringstream string_stream;

  as_decreasing_addresses(string_stream);

  return string_stream.str();
}


template <typename T>
void ToHexString<T>::increasing_addresses_print() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_uchars = reinterpret_cast<const unsigned char*>(&value_);

  // Initializer list doesn't work to make std::list have size of N.
  std::list<unsigned int> range_list (number_of_bytes);
  std::iota(range_list.begin(), range_list.end(), 0);

  std::for_each(
    range_list.begin(),
    range_list.end(),
    [&x_as_uchars](const auto& index)
    {
      printf("%01x ", x_as_uchars[index]);
    });
}

template <typename T>
void ToHexString<T>::decreasing_addresses_print() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_uchars = reinterpret_cast<const unsigned char*>(&value_);

  // Initializer list doesn't work to make std::list have size of N.
  std::list<unsigned int> range_list (number_of_bytes);
  std::generate(
    range_list.begin(),
    range_list.end(),
    [n = (number_of_bytes - 1)]() mutable
    {
      return n--;
    });

  std::for_each(
    range_list.begin(),
    range_list.end(),
    [&x_as_uchars](const auto& index)
    {
      printf("%01x ", x_as_uchars[index]);
    });
}

//------------------------------------------------------------------------------
/// \brief Prints values to \c ostream using hexadecimal notation.
///     int9_t value {10};
///     out << to_hex_string(value); // 0x0A
///     out << to_hex_string('A');   // 0x41
//------------------------------------------------------------------------------
template <typename T>
ToHexString<T> to_hex_string(const T& value)
{
  return ToHexString<T>(value);
}

namespace Detail
{

//------------------------------------------------------------------------------
/// Remaps single-byte character types into integer types to make ostream print
/// them as hex numbers.
//------------------------------------------------------------------------------
template <typename T>
  std::enable_if_t<sizeof(T) == 1, unsigned> int_value(const T x)
{
  return x & 0xff;
}

//------------------------------------------------------------------------------
/// Passes values of non-character types through without affecting them.
//------------------------------------------------------------------------------
template <typename T>
std::enable_if_t<sizeof(T) != 1, T> int_value(const T x)
{
  return x;
}

} // namespace Detail

//------------------------------------------------------------------------------
/// \brief Prints an integer value in hex.
//------------------------------------------------------------------------------
template <
  typename T,
  std::enable_if_t<std::is_integral<T>::value>* = nullptr>
std::ostream& operator<<(std::ostream& out, const ToHexString<T>& rhs)
{
  constexpr int digits_per_byte {2};
  constexpr int padding {sizeof(T) * digits_per_byte};
  auto value = Detail::int_value<T>(rhs.value_);

  return out << std::hex << std::setfill('0') << std::setw(padding) << value;
}

//------------------------------------------------------------------------------
/// \brief Prints a floating point value in hex.
//------------------------------------------------------------------------------
template <
  typename T,
  std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
std::ostream& operator<<(std::ostream& out, const ToHexString<T>& rhs)
{
  const bool is_uppercase {(out.flags() & std::ios::uppercase) != 0};
  // A a
  // C++11, converts floating-point number to hexadecimal exponent notation.
  // https://en.cppreference.com/w/cpp/io/c/fprintf
  const char* format {is_uppercase ? "%A" : "%a"};

  // "-0x1.fffffffffffffffep-16382\0" at 27 characters appears to be the largest
  // possible buffer size for long doubles. Using 40 to be safe (e.g. NaN
  // representation)
  std::array<char, 40> buffer;

  // cf. https://en.cppreference.com/w/c/io/fprintf
  // Loads the data from given locations, converts them to character string
  // equivalents and writes results to a variety of sinks.
  // int snprintf(char* restrict buffer, size_t bufsz,
  //  const char *restrict format, ...);
  // Writes results to charcter string buffer. At most bufsz - 1 characters are
  // written. Resulting character string will be terminated with null character,
  // unless bufsz is zero. If bufsz is zero, nothing is written and buffer may
  // be a null pointer, however return value (number of bytes that would be
  // written not including null terminator) is still calculated and returned.
  std::snprintf(&buffer[0], buffer.size(), format, rhs.value_);

  // This branching takes care of correctly printing "nan", "inf", "-inf",
  // std::ios::showbase's "0x", std::ios::uppercase, for negative or positive
  // numbers.

  // cf. https://en.cppreference.com/w/cpp/io/ios_base/fmtflags
  // static constexpr fmtflags showbase = /* implementation defined */
  // Specifies available formatting flags; it's a BitmaskType.
  // showbase generate a prefix indicating the numeric base for integer output,
  // require currency indicator in monetary I/O;
  if ((out.flags() & std::ios::showbase) == 0)
  {
    char letter_x {is_uppercase ? 'X' : 'x'};

    if (buffer[0] == '-' && buffer[2] == letter_x)
    {
      return out << '-' << &buffer[3];
    }

    if (buffer[1] == letter_x)
    {
      return out << &buffer[2];
    }
  }

  return out << &buffer[0];
}

} // namespace Utilities

#endif // UTILITIES_TO_HEX_STRING_H