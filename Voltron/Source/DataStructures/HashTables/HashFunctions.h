#ifndef DATA_STRUCTURES_HASH_TABLES_HASH_FUNCTIONS_H
#define DATA_STRUCTURES_HASH_TABLES_HASH_FUNCTIONS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace DataStructures
{
namespace HashTables
{

namespace HashFunctions
{

//------------------------------------------------------------------------------
/// \brief If keys are random real numbers k independently and uniformly
/// distributed in range 0 <= k < 1, then hash function is
/// h(k) = floor(km)
/// \ref pp. 262, 11.3 Hash functions, "What makes a good hash function?", 
/// Cormen, Leiserson, Rivest, and Stein (2009).
//------------------------------------------------------------------------------
class UnitIntervalToHash
{
  public:

    UnitIntervalToHash() = delete;

    UnitIntervalToHash(const std::size_t m);

    virtual ~UnitIntervalToHash() = default;

    std::size_t operator()(const double k) const;

    std::size_t get_m() const
    {
      return m_;
    }

  private:

    std::size_t m_;
};

inline std::size_t UnitIntervalToHash::operator()(const double k) const
{
  return static_cast<std::size_t>(std::floor(k * m_));
}

class DivisionMethod
{
  public:

    DivisionMethod() = delete;

    DivisionMethod(const std::size_t m);

    virtual ~DivisionMethod() = default;

    std::size_t operator()(const uint64_t k) const;

    std::size_t get_m() const
    {
      return m_;
    }

  private:

    std::size_t m_;
};

inline std::size_t DivisionMethod::operator()(const uint64_t k) const
{
  return k % m_;
}

namespace Details
{

uint64_t string_to_radix_128(const std::string& s);

} // namespace Details

} // namespace HashFunctions

} // namespace HashTables
} // namespace DataStructures

#endif // DATA_STRUCTURES_HASH_TABLES_HASH_FUNCTIONS_H