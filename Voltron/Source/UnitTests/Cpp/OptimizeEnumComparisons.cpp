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
  if (comparison_bitwise(Nombres::Un, Nombres::Un))
  {
    return 1;
  }
  else
  {
    return 0;
  }

}