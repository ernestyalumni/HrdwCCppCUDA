#ifndef UTILITIES_HEX_DUMP_H
#define UTILITIES_HEX_DUMP_H

#include <ostream>
#include <string>

namespace Utilities
{

//------------------------------------------------------------------------------
/// \ref http://www.i42.co.uk/stuff/hexdump.htm
/// \param CharT - character type for the class template basic_ostream,
/// providing character streams.
//------------------------------------------------------------------------------
template <class CharT, class Traits>
inline void hex_dump(
  const void* data,
  // length in bytes.
  std::size_t length,
  std::basic_ostream<CharT, Traits>& stream,
  std::size_t width = 16)
{
  const char* const start {static_cast<const char*>(data)};
  const char* const end {start + length};
  const char* line {start};

  while (line != end)
  {
    // Show the offset.
    stream.width(4);
    stream.fill('0');
    stream << std::hex << line - start << " : ";

    std::size_t line_length {
      std::min(width, static_cast<std::size_t>(end - line))};

    // Iterate through either Ascii or Hex print out.
    for (std::size_t pass {1}; pass <= 2; ++pass)
    {
      // Iterate through each "row" of bytes to print.
      for (const char* next {line}; next != end && next != line + width; ++next)
      {
        char ch {*next};

        switch(pass)
        {
          // Print out the ascii.
          case 1:
            stream << (ch < 32 ? '.' : ch);
            break;
          // Print out the "hex dump" of 16 bytes per line.
          case 2:
            if (next != line)
            {
              stream << " ";
            }
            stream.width(2);
            stream.fill('0');
            stream << std::hex << std::uppercase <<
              static_cast<int>(static_cast<unsigned char>(ch));
            break;
        }
      }
      if (pass == 1 && line_length != width)
      {
        // This is wrong, you want to invoke ctor of string with parentheses.
        //stream << std::string{std::to_string(width - line_length), ' '};

        // Print enough spaces, for formatting, to make enough space for final
        // bytes of the ascii printout, between the ascii and actual hex dump.
        stream << std::string(width - line_length, ' ');
      }
      
      stream << " ";
    }

    stream << std::endl;
    line = line + line_length;
  }
}

} // namespace Utilities

#endif // UTILITIES_HEX_DUMP_H
