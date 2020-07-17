//------------------------------------------------------------------------------
/// \file ClassAsEnum.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief "Enum class" but with methods (functions).
/// \details enum class cannot have methods (functions) but this does.
/// \ref https://stackoverflow.com/questions/21295935/can-a-c-enum-class-have-methods
//------------------------------------------------------------------------------
#ifndef CPP_CLASS_AS_ENUM_H
#define CPP_CLASS_AS_ENUM_H

namespace Cpp
{

class ClassAsEnum
{
  public:
  
    // enum class doesn't work here, error is
    // error: ‘Zero’ is not a member of ‘Cpp::ClassAsEnum’ in the unit test.
    enum Valeur : unsigned char
    {
      Zero,
      Un,
      Deux,
      Trois,
      Quatre
    };

    ClassAsEnum();   

    constexpr ClassAsEnum(const Valeur nombre):
      valeur_{nombre}
    {}

    // Without this, something like
    // const ClassAsEnum f0 {ClassAsEnum::Zero};
    // f0 == ClassAsEnum::Zero
    // Results in error: error: no match for ‘operator==’
    operator Valeur() const
    {
      return valeur_;
    }
    

    bool est_zero() const
    {
      return valeur_ == Valeur::Zero;
    }

    bool est_un_o_trois() const
    {
      return valeur_ == Valeur::Un || valeur_ == Valeur::Trois;
    }

    bool est_deux_o_trois() const
    {
      return valeur_ == Valeur::Deux || valeur_ == Valeur::Trois;
    }

  private:

    Valeur valeur_;
};

class ClassAsEnumClass
{
  public:
  
    // enum class doesn't work here, error is
    // error: ‘Zero’ is not a member of ‘Cpp::ClassAsEnum’ in the unit test.
    enum class Valeur : unsigned char
    {
      Zero,
      Un,
      Deux,
      Trois,
      Quatre
    };

    ClassAsEnumClass();   

    constexpr ClassAsEnumClass(const Valeur nombre):
      valeur_{nombre}
    {}

    operator Valeur() const
    {
      return valeur_;
    }

    bool est_zero() const
    {
      return valeur_ == Valeur::Zero;
    }

    bool est_un_o_trois() const
    {
      return valeur_ == Valeur::Un || valeur_ == Valeur::Trois;
    }

    bool est_deux_o_trois() const
    {
      return valeur_ == Valeur::Deux || valeur_ == Valeur::Trois;
    }

  private:

    Valeur valeur_;
};

} // namespace Cpp

#endif// CPP_CLASS_AS_ENUM_H