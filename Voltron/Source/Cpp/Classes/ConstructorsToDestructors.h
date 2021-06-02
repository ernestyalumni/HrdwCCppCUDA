//------------------------------------------------------------------------------
/// \file RuleOf5.h
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
///-----------------------------------------------------------------------------
#ifndef CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H
#define CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H

#include <cstddef> // std::size_T
#include <initializer_list>
#include <string>

namespace Cpp
{
namespace Classes
{

class DefaultConstructs
{
  public:

    //--------------------------------------------------------------------------
    /// \details Classes that have custom ctors, dtors should deal exclusively
    /// with ownership (following Single Responsibility Principle). Rule of 0
    /// also appears in C++ Core Guideline C.20, If you can avoid defining
    /// default operations, do.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    /// \brief Default Constructor.
    /// \details Default constructor does not allow for members that are
    /// references since they can't be initialized with r-values (cannot bind
    /// non-const lvalue reference of type 'T&' to rvalue of type T).
    //--------------------------------------------------------------------------
    DefaultConstructs();

    //--------------------------------------------------------------------------
    /// \brief Custom Constructor.
    /// \details Try Rule of 0, not defining its copy and move.
    //--------------------------------------------------------------------------
    DefaultConstructs(const std::string& s, const int value);

    virtual ~DefaultConstructs();

    static int default_ctor_counter()
    {
      return default_ctor_counter_;
    }

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    //--------------------------------------------------------------------------
    /// \details Try returning a reference.
    //--------------------------------------------------------------------------    
    std::string& s_data()
    {
      return s_data_;
    }

    int int_data() const
    {
      return int_data_;
    }

    void int_data(const int change_value)
    {
      int_data_ = change_value;
    }

    // Try without the const qualifier.
    bool is_default_constructed()
    {
      return is_default_constructed_;
    }

    //--------------------------------------------------------------------------
    /// \ref https://stackoverflow.com/questions/1563897/static-constant-string-class-member
    /// \details Syntax with inline needed for C++17.
    /// TODO: With C++17, possibly could use inline instead for something like:
    /// inline static const std::string dtor_message =
    ///   "DefaultConstructs destructs"s;
    //--------------------------------------------------------------------------    
    static const std::string default_ctor_message;
    static const std::string dtor_message;

  private:

    std::string s_data_;
    int int_data_;

    bool is_default_constructed_{false};

    //--------------------------------------------------------------------------
    /// \ref Gottschling (2015), 2.2.4 The Static Declarator for Classes
    /// https://stackoverflow.com/questions/9110487/undefined-reference-to-a-static-member
    /// \details Member variables declared static exist only once per class.
    /// This allows us to share a resource between objects of a class. Another
    /// use case is for creating a Singleton.
    ///
    /// You need to define in .cpp.
    //--------------------------------------------------------------------------
    static int default_ctor_counter_;
    static int dtor_counter_;
};

class NoDefaultConstruction
{
  public:

    NoDefaultConstruction() = delete;

    NoDefaultConstruction(int& int_ref, const int data = 0);

    virtual ~NoDefaultConstruction();

    static int ctor_1_counter()
    {
      return ctor_1_counter_;
    }

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    int int_ref() const
    {
      return int_ref_;
    }

    void increment_int_ref(const int value = 1)
    {
      int_ref_ += value;
    }

    int& int_data()
    {
      return int_data_;
    }

    static const std::string ctor_1_message;
    static const std::string dtor_message;

  private:

    //--------------------------------------------------------------------------
    /// \ref 2.3.1 Constructors, Gottschling (2015), consider what happens for
    /// references.
    //--------------------------------------------------------------------------
    int& int_ref_;

    int int_data_;

    static int ctor_1_counter_;
    static int dtor_counter_;
};

NoDefaultConstruction return_no_default_construction_as_type(
  int& int_ref,
  const int data);

class CopyConstructOnly
{
  public:

    CopyConstructOnly() = delete;

    CopyConstructOnly(
      const int value,
      const std::size_t N,
      std::initializer_list<int> int_list);

    // Copy construction.
    CopyConstructOnly(const CopyConstructOnly&);

    // Copy assignment.
    CopyConstructOnly& operator=(const CopyConstructOnly&) = delete;

    // Move ctor.
    CopyConstructOnly(CopyConstructOnly&&) = delete;
    // Move assignment.
    CopyConstructOnly& operator=(CopyConstructOnly&&) = delete;

    virtual ~CopyConstructOnly();

    static int ctor_1_counter()
    {
      return ctor_1_counter_;
    }

    static int copy_ctor_counter()
    {
      return copy_ctor_counter_;
    }

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    static const std::string ctor_1_message;
    static const std::string copy_ctor_message;
    static const std::string dtor_message;

  private:

    int int_data_;
    std::size_t N_;
    int* array_;

    static int ctor_1_counter_;
    static int copy_ctor_counter_;
    static int dtor_counter_;
};

class MoveOnlyLight
{
  public:

    MoveOnlyLight() = delete;

    MoveOnlyLight(const int value);

    // Copy ctor.
    MoveOnlyLight(const MoveOnlyLight&) = delete;
    // Copy assignment.
    MoveOnlyLight& operator=(const MoveOnlyLight&) = delete;

    // Move ctor.
    MoveOnlyLight(MoveOnlyLight&&);

    // Move assignment.
    MoveOnlyLight& operator=(MoveOnlyLight&&);

    virtual ~MoveOnlyLight();

    static int ctor_counter()
    {
      return ctor_counter_;
    }

    static int move_ctor_counter()
    {
      return move_ctor_counter_;
    }

    static int move_assign_counter()
    {
      return move_assign_counter_;
    }

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    static const std::string ctor_message;
    static const std::string move_ctor_message;
    static const std::string move_assign_message;
    static const std::string dtor_message;

    int data() const
    {
      return data_;
    }

  private:

    int data_;

    static int ctor_counter_;
    static int move_ctor_counter_;
    static int move_assign_counter_;
    static int dtor_counter_;
};

MoveOnlyLight return_rvalue_move_only_light(const int value);

class CustomDestructorLight
{
  public:

    CustomDestructorLight() = delete;

    CustomDestructorLight(const int value);

    // Copy ctor.
    CustomDestructorLight(const CustomDestructorLight&) = delete;
    // Copy assignment.
    CustomDestructorLight& operator=(const CustomDestructorLight&) = delete;

    // Move ctor.
    CustomDestructorLight(CustomDestructorLight&&);

    // Move assignment.
    CustomDestructorLight& operator=(CustomDestructorLight&&) = delete;

    virtual ~CustomDestructorLight();

    static int move_ctor_counter()
    {
      return move_ctor_counter_;
    }

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    static const std::string move_ctor_message;
    static const std::string dtor_message;

    int data() const
    {
      return data_;
    }

    void data(const int input)
    {
      data_ = input;
    }

  private:

    int data_;

    static int move_ctor_counter_;
    static int dtor_counter_;
};


class CustomDestructorEncapsulated
{
  public:

    CustomDestructorEncapsulated() = delete;

    CustomDestructorEncapsulated(const int value1, const int value2);

    // Copy ctor.
    CustomDestructorEncapsulated(const CustomDestructorEncapsulated&) = delete;
    // Copy assignment.
    CustomDestructorEncapsulated& operator=(
      const CustomDestructorEncapsulated&) = delete;

    // Move ctor.
    CustomDestructorEncapsulated(CustomDestructorEncapsulated&&) = delete;

    // Move assignment.
    CustomDestructorEncapsulated& operator=(
      CustomDestructorEncapsulated&&) = delete;

    virtual ~CustomDestructorEncapsulated();

    static int dtor_counter()
    {
      return dtor_counter_;
    }

    static const std::string dtor_message;

    int data() const
    {
      return data_.data();
    }

    void data(const int input)
    {
      data_.data(input);
    }

    int other_data() const
    {
      return other_data_;      
    }

    void other_data(const int input)
    {
      other_data_ = input;
    }

  private:

    CustomDestructorLight data_;
    int other_data_;

    static int dtor_counter_;
};

} // namespace Classes
} // namespace Cpp

#endif // CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H