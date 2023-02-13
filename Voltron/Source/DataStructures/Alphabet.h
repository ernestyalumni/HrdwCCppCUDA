#ifndef DATA_STRUCTURES_ALPHABET_H
#define DATA_STRUCTURES_ALPHABET_H

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

namespace DataStructures
{

//-----------------------------------------------------------------------------
/// See https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/Alphabet.java
/// See Algorithms, 4th. Ed., Robert Sedgewick, Kevin Wayne.
//-----------------------------------------------------------------------------

class Alphabet
{
	public:

    // In Java, Character.Max is 2^16 = 65536
    inline static constexpr std::size_t character_maximum_ {
      std::numeric_limits<char16_t>::max()};

    Alphabet();

    Alphabet(const std::string& alphabet);

    ~Alphabet();

  private:

    // The characters.
    char* alphabet_;
    // indices.
    std::optional<std::size_t>* inverse_;
    // The radix of the alphabet.
    std::size_t r_;
};

} // namespace DataStructures

#endif // DATA_STRUCTURES_ALPHABET_H