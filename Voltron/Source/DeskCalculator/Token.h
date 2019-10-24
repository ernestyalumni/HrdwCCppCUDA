//------------------------------------------------------------------------------
/// \file Token.h
/// \author Ernest Yeung
/// \brief Desk calculator demonstrating C++ expressions.
///-----------------------------------------------------------------------------
#ifndef _DESK_CALCULATOR_TOKEN_H_
#define _DESK_CALCULATOR_TOKEN_H_
  
#include <string>

namespace DeskCalculator
{
namespace Token
{

enum class Kind : char
{
  name,
  number,
  end,
  plus = '+',
  minus = '-',
  mul = '*',
  div = '/',
  print=';',
  assign='=',
  lp='(',
  rp=')'
};

struct Token
{
  Kind kind_;
  std::string string_value_;
  double number_value_;
};

//------------------------------------------------------------------------------
/// \class TokenStream
/// \brief 
/// \details Stroustrup, The C++ Programming Language (2013), pp. 247, 10.2.2
/// Input. 
//------------------------------------------------------------------------------
class TokenStream
{
  public:

    TokenStream(std::istream& s);
    TokenStream(std::istream* p);

    ~TokenStream();

    Token get(); // read and return next token
    const Token& current(); // most recently read token

    void set_input(std::istream& s);
    void set_input(std::istream* p);

  private:
};

} // namespace Token
} // namespace DeskCalculator


#endif // _DESK_CALCULATOR_TOKEN_H_