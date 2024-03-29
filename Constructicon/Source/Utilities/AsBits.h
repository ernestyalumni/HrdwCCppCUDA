#ifndef UTILITIES_AS_BITS_H
#define UTILITIES_AS_BITS_H

#include <bitset>
#include <climits>
#include <iomanip> // std::setw
#include <ios>
#include <sstream>
#include <string>
#include <type_traits>

namespace Utilities
{

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
class AsBits : public std::bitset<sizeof(T) * CHAR_BIT>
{
  public:

    using std::bitset<sizeof(T) * CHAR_BIT>::bitset;
    using std::bitset<sizeof(T) * CHAR_BIT>::to_ullong;
    using std::bitset<sizeof(T) * CHAR_BIT>::to_ulong;

    AsBits(const AsBits& other):
      std::bitset<sizeof(T) * CHAR_BIT>{other.to_string()}
    {}

    AsBits& operator=(const AsBits& other)
    {
      return AsBits<T>{other.to_string()};
    }

    T to_integer() const
    {
      return sizeof(T) <= sizeof(unsigned int) ? static_cast<T>(to_ulong()) :
        static_cast<T>(to_ullong());
    }

    std::string as_hex_string() const
    {
      std::stringstream ss;

      T value {to_integer()};

      unsigned char* value_ptr {
        static_cast<unsigned char*>(static_cast<void*>(&value))};

      for (std::size_t i {sizeof(T)}; i > 0; --i)
      {
        ss << std::hex <<
          std::setw(2) <<
          std::setfill('0') <<
          static_cast<int>(value_ptr[i - 1]);
      }

      return ss.str();
    }
};

} // namespace Utilities

#endif // UTILITIES_AS_BITS_H