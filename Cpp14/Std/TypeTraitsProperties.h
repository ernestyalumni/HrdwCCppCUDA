//------------------------------------------------------------------------------
/// \file TypeTraitsProperties.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Print out all type_traits properties of a template parameter T.
/// \ref https://en.cppreference.com/w/cpp/header/type_traits
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
///
/// \details
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 List_main.cpp -o List_main
//------------------------------------------------------------------------------
#ifndef STD_TYPE_TRAITS_PROPERTIES_H
#define STD_TYPE_TRAITS_PROPERTIES_H

#include <iostream>
#include <type_traits>

namespace Std
{

template <typename T>
class PrimaryTypeTraits
{
  public:

    PrimaryTypeTraits():
      is_void_{std::is_void<T>::value},
      is_null_pointer_{std::is_null_pointer<T>::value},
      is_integral_{std::is_integral<T>::value},
      is_floating_point_{std::is_floating_point<T>::value},
      is_array_{std::is_array<T>::value},
      is_enum_{std::is_enum<T>::value},
      is_union_{std::is_union<T>::value},
      is_class_{std::is_class<T>::value},
      is_function_{std::is_function<T>::value},
      is_pointer_{std::is_pointer<T>::value},
      is_lvalue_reference_{std::is_lvalue_reference<T>::value},
      is_rvalue_reference_{std::is_rvalue_reference<T>::value},
      is_member_object_pointer_{std::is_member_object_pointer<T>::value},
      is_member_function_pointer_{std::is_member_function_pointer<T>::value}
    {}

    bool is_void() const { return is_void_; }
    bool is_null_pointer() const { return is_null_pointer_; }
    bool is_integral() const { return is_integral_; }
    bool is_floating_point() const { return is_floating_point_; }
    bool is_array() const { return is_array_; }
    bool is_enum() const { return is_enum_; }
    bool is_union() const { return is_union_; }
    bool is_class() const { return is_class_; }
    bool is_function() const { return is_function_; }
    bool is_pointer() const { return is_pointer_; }
    bool is_lvalue_reference() const { return is_lvalue_reference_; }
    bool is_rvalue_reference() const { return is_rvalue_reference_; }
    bool is_member_object_pointer() const { return is_member_object_pointer_; }
    bool is_member_function_pointer() const
    {
      return is_member_function_pointer_;
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os,
      const PrimaryTypeTraits<U>&);

  private:

    const bool is_void_;
    const bool is_null_pointer_;
    const bool is_integral_;
    const bool is_floating_point_;
    const bool is_array_;
    const bool is_enum_;
    const bool is_union_;
    const bool is_class_;
    const bool is_function_;
    const bool is_pointer_;
    const bool is_lvalue_reference_;
    const bool is_rvalue_reference_;
    const bool is_member_object_pointer_;
    const bool is_member_function_pointer_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const PrimaryTypeTraits<T>& traits)
{
  os << " is_void : " << traits.is_void() <<
    " is_null_pointer : " << traits.is_null_pointer() <<
    " is_integral : " << traits.is_integral() <<
    " is_floating_point : " << traits.is_floating_point() <<
    " is_array : " << traits.is_array() <<
    " is_enum : " << traits.is_enum() <<
    " is_union : " << traits.is_union() <<
    " is_class : " << traits.is_class() <<
    " is_function : " << traits.is_function() <<
    " is_pointer : " << traits.is_pointer() <<
    " is_lvalue_reference : " << traits.is_lvalue_reference() <<
    " is_rvalue_reference : " << traits.is_rvalue_reference() <<
    " is_member_object_pointer : " << traits.is_member_object_pointer() <<
    " is_member_function_pointer : " << traits.is_member_function_pointer() <<
      '\n';

  return os;
}

template <typename T>
class CompositeTypeTraits
{
  public:

    CompositeTypeTraits():
      is_fundamental_{std::is_fundamental<T>::value},
      is_arithmetic_{std::is_arithmetic<T>::value},
      is_scalar_{std::is_scalar<T>::value},
      is_object_{std::is_object<T>::value},
      is_compound_{std::is_compound<T>::value},
      is_reference_{std::is_reference<T>::value},
      is_member_pointer_{std::is_member_pointer<T>::value}
    {}

    bool is_fundamental() const { return is_fundamental_; }
    bool is_arithmetic() const { return is_arithmetic_; }
    bool is_scalar() const { return is_scalar_; }
    bool is_object() const { return is_object_; }
    bool is_compound() const { return is_compound_; }
    bool is_reference() const { return is_reference_; }
    bool is_member_pointer() const { return is_member_pointer_; }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os,
      const CompositeTypeTraits<U>&);

  private:

    const bool is_fundamental_;
    const bool is_arithmetic_;
    const bool is_scalar_;
    const bool is_object_;
    const bool is_compound_;
    const bool is_reference_;
    const bool is_member_pointer_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const CompositeTypeTraits<T>& traits)
{
  os << " is_fundamental : " << traits.is_fundamental() <<
    " is_arithmetic : " << traits.is_arithmetic() <<
    " is_scalar : " << traits.is_scalar() <<
    " is_object : " << traits.is_object() <<
    " is_compound : " << traits.is_compound() <<
    " is_reference : " << traits.is_reference() <<
    " is_member_pointer : " << traits.is_member_pointer() << '\n';

  return os;
}

template <typename T>
class TypeProperties
{
  public:

    TypeProperties():
      is_const_{std::is_const<T>::value},
      is_volatile_{std::is_volatile<T>::value},
      is_trivial_{std::is_trivial<T>::value},
      is_trivially_copyable_{std::is_trivially_copyable<T>::value},
      is_standard_layout_{std::is_standard_layout<T>::value},
//      has_unique_object_representations_{ // C++17
  //      std::has_unique_object_representations<T>::value}, // C++17
      is_empty_{std::is_empty<T>::value},
      is_polymorphic_{std::is_polymorphic<T>::value},
      is_abstract_{std::is_abstract<T>::value},
      is_final_{std::is_final<T>::value},
//      is_aggregate_{std::is_aggregate<T>::value}, // C++17
      is_signed_{std::is_signed<T>::value},
      is_unsigned_{std::is_unsigned<T>::value}
    {}

    bool is_const() const { return is_const_; }
    bool is_volatile() const { return is_volatile_; }
    bool is_trivial() const { return is_trivial_; }
    bool is_trivially_copyable() const { return is_trivially_copyable_; }
    bool is_standard_layout() const { return is_standard_layout_; }
//    bool has_unique_object_representations() const // C++17
//    { // C++17
  //    return has_unique_object_representations_; // C++17
    //} // C++17
    bool is_empty() const { return is_empty_; }
    bool is_polymorphic() const { return is_polymorphic_; }
    bool is_abstract() const { return is_abstract_; }
    bool is_final() const { return is_final_; }
//    bool is_aggregate() const { return is_aggregate_; } // C++17
    bool is_signed() const { return is_signed_; }
    bool is_unsigned() const { return is_unsigned_; }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os,
      const TypeProperties<U>&);

  private:

    const bool is_const_;
    const bool is_volatile_;
    const bool is_trivial_;
    const bool is_trivially_copyable_;
    const bool is_standard_layout_;
//    const bool has_unique_object_representations_; // C++17
    const bool is_empty_;
    const bool is_polymorphic_;
    const bool is_abstract_;
    const bool is_final_;
//    const bool is_aggregate_; // C++17
    const bool is_signed_;
    const bool is_unsigned_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const TypeProperties<T>& traits)
{
  os << " is_const : " << traits.is_const() <<
    " is_volatile : " << traits.is_volatile() <<
    " is_trivial : " << traits.is_trivial() <<
    " is_trivially_copyable : " << traits.is_trivially_copyable() <<
    " is_standard_layout : " << traits.is_standard_layout() <<
//    " has_unique_object_representations : " << // C++17
//      traits.has_unique_object_representations() << // C++17
    " is_empty : " << traits.is_empty() <<
    " is_polymorphic : " << traits.is_polymorphic() <<
    " is_abstract : " << traits.is_abstract() <<
    " is_final : " << traits.is_final() <<
//    " is_aggregate : " << traits.is_aggregate() << // C++17
    " is_signed : " << traits.is_signed() <<
    " is_unsigned : " << traits.is_unsigned() << '\n';

  return os;
}

template <typename T>
class RuleOf5Properties
{
  public:

    RuleOf5Properties():
      is_constructible_{std::is_constructible<T>::value},
      is_trivially_constructible_{std::is_trivially_constructible<T>::value},
      is_nothrow_constructible_{std::is_nothrow_constructible<T>::value},
      is_default_constructible_{std::is_default_constructible<T>::value},
      is_trivially_default_constructible_{
        std::is_trivially_default_constructible<T>::value},
      is_nothrow_default_constructible_{
        std::is_nothrow_default_constructible<T>::value},
      is_copy_constructible_{std::is_copy_constructible<T>::value},
      is_trivially_copy_constructible_{
        std::is_trivially_copy_constructible<T>::value},
      is_nothrow_copy_constructible_{
        std::is_nothrow_copy_constructible<T>::value},
      is_move_constructible_{std::is_move_constructible<T>::value},
      is_trivially_move_constructible_{
        std::is_trivially_move_constructible<T>::value},
      is_nothrow_move_constructible_{
        std::is_nothrow_move_constructible<T>::value},
//      is_assignable_{std::is_assignable<T>::value},
//      is_trivially_assignable_{std::is_trivially_assignable<T>::value},
//      is_nothrow_assignable_{std::is_nothrow_assignable<T>::value},
      is_copy_assignable_{std::is_copy_assignable<T>::value},
      is_trivially_copy_assignable_{
        std::is_trivially_copy_assignable<T>::value},
      is_nothrow_copy_assignable_{
        std::is_nothrow_copy_assignable<T>::value},
      is_move_assignable_{std::is_move_assignable<T>::value},
      is_trivially_move_assignable_{
        std::is_trivially_move_assignable<T>::value},
      is_nothrow_move_assignable_{std::is_nothrow_move_assignable<T>::value},
      is_destructible_{std::is_destructible<T>::value},
      is_trivially_destructible_{std::is_trivially_destructible<T>::value},
      is_nothrow_destructible_{std::is_nothrow_destructible<T>::value},
      has_virtual_destructor_{std::has_virtual_destructor<T>::value}
    {}

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os,
      const RuleOf5Properties<U>&);

  private:

    const bool is_constructible_;
    const bool is_trivially_constructible_;
    const bool is_nothrow_constructible_;
    const bool is_default_constructible_;
    const bool is_trivially_default_constructible_;
    const bool is_nothrow_default_constructible_;
    const bool is_copy_constructible_;
    const bool is_trivially_copy_constructible_;
    const bool is_nothrow_copy_constructible_;
    const bool is_move_constructible_;
    const bool is_trivially_move_constructible_;
    const bool is_nothrow_move_constructible_;
//    const bool is_assignable_;
//    const bool is_trivially_assignable_;
//    const bool is_nothrow_assignable_;
    const bool is_copy_assignable_;
    const bool is_trivially_copy_assignable_;
    const bool is_nothrow_copy_assignable_;
    const bool is_move_assignable_;
    const bool is_trivially_move_assignable_;
    const bool is_nothrow_move_assignable_;
    const bool is_destructible_;
    const bool is_trivially_destructible_;
    const bool is_nothrow_destructible_;
    const bool has_virtual_destructor_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const RuleOf5Properties<T>& traits)
{
  os << 
    " is_constructible : " << traits.is_constructible() <<
    " is_trivially_constructible : " << traits.is_trivially_constructible() <<
    " is_nothrow_constructible : " << traits.is_nothrow_constructible() <<
    " is_default_constructible : " << traits.is_default_constructible() <<
    " is_trivially_default_constructible : " <<
      traits.is_trivially_default_constructible() <<
    " is_nothrow_default_constructible : " <<
      traits.is_nothrow_default_constructible() <<
    " is_copy_constructible : " << traits.is_copy_constructible() <<
    " is_trivially_copy_constructible : " <<
      traits.is_trivially_copy_constructible() <<
    " is_nothrow_copy_constructible : " << traits.is_nothrow_copy_constructible() <<
    " is_move_constructible : " << traits.is_move_constructible() <<
    " is_trivially_move_constructible : " <<
      traits.is_trivially_move_constructible() <<
    " is_nothrow_move_constructible : " << traits.is_nothrow_move_constructible() <<
//    " is_assignable : " << traits.is_assignable() <<
//    " is_trivially_assignable : " << traits.is_trivially_assignable() <<
//    " is_nothrow_assignable : " << traits.is_nothrow_assignable() <<
    " is_copy_assignable : " << traits.is_copy_assignable() <<
    " is_trivially_copy_assignable : " << traits.is_copy_trivially_assignable() <<
    " is_nothrow_copy_assignable : " << traits.is_nothrow_copy_assignable() <<
    " is_move_assignable : " << traits.is_move_assignable() <<
    " is_trivially_move_assignable : " <<
      traits.is_trivially_move_assignable() <<
    " is_nothrow_move_assignable : " << traits.is_nothrow_move_assignable() <<
    " is_destructible : " << traits.is_destructible() <<
    " is_trivially_destructible : " << traits.is_trivially_destructible() <<
    " is_nothrow_destructible : " << traits.is_nothrow_destructible() <<
    " has_virtual_destructor : " << traits.has_virtual_destructor() << '\n';

  return os;
}

} // namespace Std

#endif // STD_TYPE_TRAITS_PROPERTIES_H
