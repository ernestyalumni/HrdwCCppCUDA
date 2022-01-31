#include "HashFunctions.h"

#include <algorithm> // std::for_each
#include <cstddef>
#include <cstdint>
#include <string>

namespace DataStructures
{
namespace HashTables
{

namespace HashFunctions
{

UnitIntervalToHash::UnitIntervalToHash(const std::size_t m):
  m_{m}
{}

DivisionMethod::DivisionMethod(const std::size_t m):
  m_{m}
{}

namespace Details
{

uint64_t string_to_radix_128(const std::string& s)
{
  std::size_t counter {s.size() - 1};
  uint64_t sum {0};

  std::for_each(
    s.cbegin(),
    s.cend(),
    [&counter, &sum](const char c) {
      sum += std::pow(128, counter) * static_cast<uint64_t>(c);
      --counter;
    });

  return sum;
}

} // namespace Details

} // namespace HashFunctions

} // namespace HashTables
} // namespace DataStructures
