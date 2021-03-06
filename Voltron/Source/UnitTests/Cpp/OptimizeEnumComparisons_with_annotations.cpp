//------------------------------------------------------------------------------
/// \file OptimizeEnumComparison_with_annotations.cpp
/// \details 
/// 
/// cf. https://www.cs.umd.edu/~srhuang/teaching/cmsc212/gdb-tutorial-handout.pdf
/// Compilation examples:
/// g++ -I../../ -Wall -Werror -ansi -pedantic-errors -g -std=c++2a OptimizeEnumComparisons.cpp -o OptimizeEnumComparisons
/// -g is for enabling built-in debugging support (which gdb needs).
/// -std=c++2a is after -g in order for the compiler to recognize the flag
/// (otherwise, error is obtained saying C++11 is needed).
/// 
/// gdb
/// gdb OptimizeEnumComparisons // Specified program to debug
///
/// Assembler Code generated by GCC
/// https://panthema.net/2013/0124-GCC-Output-Assembler-Code/
/// Investigate compiler output.
/// gcc internally writes assembler code, which then is translated into binary
/// machine code.
/// Compilation example:
/// g++ -I../../ -Wall -Werror -ansi -pedantic-errors -std=c++2a OptimizeEnumComparisons.cpp -o OptimizeEnumComparisons -Wa,-adhln=OptimizeEnumComparisons.s -g -fverbose-asm -masm=intel
/// less flags
/// g++ -I../../ -std=c++2a OptimizeEnumComparisons.cpp -o OptimizeEnumComparisons -Wa,-adhln=OptimizeEnumComparisons_less_flags.s -g -fverbose-asm -masm=intel
/// -Wa, -adhln=test.s additional compiler option instructs gcc to pass
/// additional options to internally called assembler:"-adhln=test.s". These
/// tell as to output a listing to test.s according to following parameters.
/// debug -g interleaves assembler listing with original code.
/// -fverbose-asm, gcc ouputs some additional info about which variable is
/// manipulated in a register.
/// -masm=intel changes assembler mnemonics to Intel's style, instead of AT&T
/// style; Intel's style is right-to-left assignment paradigm, resembling C
/// assignment.
/// https://stackoverflow.com/questions/24787769/what-are-lfb-lbb-lbe-lvl-loc-in-the-compiler-generated-assembly-code
//------------------------------------------------------------------------------

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

using Cpp::Utilities::TypeSupport::get_underlying_value;

enum class Nombres
{
  Zero,
  Un,
  Deux,
  Trois,
  Quatr 
};

inline bool comparison_bitwise(const Nombres lhs, const Nombres rhs)
{
  return
    static_cast<bool>(get_underlying_value<const Nombres>(lhs) &
      get_underlying_value<const Nombres>(rhs));
}

inline bool comparison_3_way(const Nombres lhs, const Nombres rhs) 
{
  return (lhs == Nombres::Trois) || (lhs == rhs);
}

int main()
{
  
}