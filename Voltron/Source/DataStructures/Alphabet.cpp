#include "Alphabet.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace DataStructures
{

Alphabet::Alphabet(const std::string& alphabet):
  alphabet_{new char[alphabet.length() + 1]},
  inverse_{new std::optional<std::size_t>[character_maximum_](std::nullopt)},
  r_{alphabet.length()}
{
  // Check that alphabet contains no duplicate characters.
  std::array<bool, character_maximum_> is_seen {false};
  std::fill(is_seen.begin(), is_seen.end(), false);
  for (char ch : alphabet)
  {
    const std::size_t index {static_cast<std::size_t>(ch)};

    if (is_seen[index])
    {
      throw std::runtime_error("Alphabet input contains duplicate characters");
    }

    is_seen[index] = true;
  }

  // See https://stackoverflow.com/questions/13294067/how-to-convert-string-to-char-array-in-c
  std::strcpy(alphabet_, alphabet.c_str());

  for (std::size_t i {0}; i < r_; ++i)
  {
    inverse_[static_cast<std::size_t>(alphabet[i])] = i;
  }
}

Alphabet::~Alphabet()
{
  delete [] alphabet_;
  delete [] inverse_;
}

} // namespace DataStructures
