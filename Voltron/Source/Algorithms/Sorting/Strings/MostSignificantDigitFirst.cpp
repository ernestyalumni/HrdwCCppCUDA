#include "Algorithms/Sorting/Strings/MostSignificantDigitFirst.h"

#include <algorithm>
#include <cstddef>
#include <string>

using std::size_t;
using std::string;

namespace Algorithms
{
namespace Sorting
{
namespace Strings
{

namespace MostSignificantDigitFirst
{

int char_at(const string& s, const size_t d)
{
  return d < s.length() ? static_cast<int>(s[d]) : -1;
}

bool is_less(const string& v, const string& w, const size_t d)
{
  for (std::size_t i {d}; i < std::min(v.length(), w.length()); ++i)
  {
    if (char_at(v, i) < char_at(w, i))
    {
      return true;
    }

    if (char_at(v, i) > char_at(w, i))
    {
      return false;
    }
  }

  return v.length() < w.length();
}


} // namespace MostSignificantDigitFirst

} // namespace Strings
} // namespace Sorting
} // namespace Algorithms