//------------------------------------------------------------------------------
/// \file Singletons.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Examples of Singleton idiom (thus, named Singletons)
/// \ref 2.1 Compile-Time Assertions. Ch. 2 Techniques. pp. 19
///   Andrei Alexandrescu. Modern C++: Design Generic Programming and Design
///   Patterns Applied.
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Singletons.h -o Singletons
//------------------------------------------------------------------------------
#ifndef _SINGLETONS_H_
#define _SINGLETONS_H_

#include <memory> // std::make_unique
#include <mutex> // std::once_flag, std::call_once
#include <string> // std::string, for demonstration purposes.

namespace Idioms
{

namespace Singletons
{

template <class T>
class SingletonT
{
  public:

    explicit SingletonT<T>(const std::string& name):
      name_{name}
    {}

    SingletonT<T>() = delete;
    SingletonT<T>(SingletonT<T> const&) = delete;       // Copy construct
    SingletonT<T>(SingletonT<T>&&) = delete;            // Move construct
    SingletonT<T>& operator=(SingletonT<T> const&) = delete; // Copy assign
    SingletonT<T>& operator=(SingletonT<T>&&) = delete; // Move assign

    virtual ~SingletonT<T>() = default;

    static SingletonT<T>& get_instance()
    {
      std::call_once(once_,
        []()
        {
          instance_.reset(new SingletonT<T>("SingletonT"));
        });
    }

  private:
    const std::string name_;
    static std::unique_ptr<SingletonT<T>> instance_;
    static std::once_flag once_;
}; 

// \ref https://stackoverflow.com/questions/42181645/what-is-wrong-with-my-code-that-creates-a-static-data-member
// \ref https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
// \ref https://en.cppreference.com/w/cpp/thread/call_once
// \ref https://en.cppreference.com/w/cpp/memory/unique_ptr/reset
template <class T>
std::unique_ptr<SingletonT<T>> SingletonT<T>::instance_;

template <class T>
std::once_flag SingletonT<T>::once_;

/*
template <class T>
SingletonT<T>& SingletonT<T>::get_instance()
{
  std::call_once(once_,
    []()
    {
      instance_.reset(new SingletonT<T>("SingletonT"));
    });
}
*/

//template <>
//class SingletonT<int>;

class Singleton
{
  public:

    explicit Singleton(const std::string& name):
      name_{name}
    {}

    Singleton() = delete;
    Singleton(Singleton const&) = delete;       // Copy construct
    Singleton(Singleton&&) = delete;            // Move construct
    Singleton& operator=(Singleton const&) = delete; // Copy assign
    Singleton& operator=(Singleton&&) = delete; // Move assign

    virtual ~Singleton() = default;

    static Singleton& get_instance()
    {
      static Singleton instance {"Singleton"};
      return instance;
    }

  private:
    std::string name_;
};

//------------------------------------------------------------------------------
/// \brief Classic lazy evaluated and correctly destroyed singleton.
/// \ref https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
//------------------------------------------------------------------------------
class LazySingleton  
{
  public:
    static LazySingleton& get_instance()
    {
      static LazySingleton instance;  // Guaranteed to be destroyed.
                                      // Instantiated on first use.
      return instance;
    }

    LazySingleton(LazySingleton const&) = delete;
    void operator=(LazySingleton const&) = delete;

  private:
    LazySingleton()                  // ctor? (the {} brackets) are needed here.
    {
      // Constructor code goes here.
    }
};

//------------------------------------------------------------------------------
/// \brief "double-check, single-lock" pattern in C++11
/// \ref http://preshing.com/20130930/double-checked-locking-is-fixed-in-cpp11/
/// \ref https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
//------------------------------------------------------------------------------

class LazySingleton11
{
  public:
    static LazySingleton11& get_instance()
    {
      // Since it's a static variable, if the class has already been created,
      // it won't be created again.
      // And it **is** thread-safe in C++11
      static LazySingleton11 instance;

      // Return a reference to our instance.
      return instance;
    }

    // delete copy and move constructors and assing operators.
    LazySingleton11(LazySingleton11 const&) = delete;       // Copy construct
    LazySingleton11(LazySingleton11&&) = delete;            // Move construct
    LazySingleton11& operator=(LazySingleton11 const&) = delete; // Copy assign
    LazySingleton11& operator=(LazySingleton11&&) = delete; // Move assign

  protected:
    LazySingleton11() 
    {
      // Constructor code goes here.
    }

    ~LazySingleton11()
    {
      // Destructor code goes here.
    }
};

//------------------------------------------------------------------------------
/// \brief Singleton class example that simply stores a single string.
/// \ref https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns#Visitor
//------------------------------------------------------------------------------
/*class StringSingleton
{
  public:

    // Note that the next two functions are not given bodies, thus any attempt
    // to call them implicitly will return as compiler errors. This prevents
    // accidental copying of the only instance of the class.
    StringSingleton(const StringSingleton&) = delete; // disallow copy ctor
    StringSingleton &operator=(const StringSingleton&) = delete; // disallow
    // assignment operator

    // Accessor functions for class itself.
    std::string get_string() const
    {
      return string_;
    }

    void set_string(const std::string& new_string)
    {
      string_ = new_string;
    }

    // The magic function, which allows access to the class from anywhere.
    // To get the value of the instance of the class, call:
    //  StringSingleton::instance().get_string();
    static StringSingleton& instance()
    {
      // This line only runs once, thus creating the only instance in
      // existence.
      static std::unique_ptr<StringSingleton> instance_ptr {
        std::make_unique<StringSingleton>()
      };

      // error: invalid initialization of non-const reference of type
      // ‘Idioms::Singletons::StringSingleton&’ from an rvalue of type 
      // ‘std::unique_ptr<Idioms::Singletons::StringSingleton>::pointer 
      // {aka Idioms::Singletons::StringSingleton*}’
      // return instance_ptr.get(); // always returns the same instance
      return *instance_ptr;
    }

  private:
    // We need to make some given functions private to finish the definition of
    // the singleton

    // default ctor available only to members or friends of this class.
    StringSingleton()
    {} 

    // Note that although this should be allowed, some compilers may not 
    // implement private destructors/
    // This prevents others from deleting our one single instance, which was
    // otherwise created on the heap.
    ~StringSingleton() {}

    // private data for an instance of this class. 
    std::string string_;
};
*/

//------------------------------------------------------------------------------
/// \brief Implementation has a memory leak.
/// \ref https://gist.github.com/pazdera/1098119
//------------------------------------------------------------------------------

class BadSingleton
{
  private:
    // Here will be the instance stored.
    static BadSingleton* instance;

    // Private constructor to prevent instancing.
    BadSingleton();

  public:
    // Static access method.
    static BadSingleton* get_instance();
};


BadSingleton::BadSingleton()
{}

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
//------------------------------------------------------------------------------
template <class T>
class SingletonResource
{
  public:
    virtual ~SingletonResource<T>() = default;
    static SingletonResource<T>& get_instance()
    {
      std::call_once(once_,
        []()
        {
          instance_.reset(new T);
        });
      return *instance_.get();
    }

    SingletonResource<T>(const SingletonResource<T>&) = delete;
    SingletonResource<T>& operator=(const SingletonResource<T>&) = delete;

  private:

    static std::unique_ptr<SingletonResource<T>> instance_;
    static std::once_flag once_;

    SingletonResource<T>() = default;
};

} // namespace Singletons

} // namespace Idioms

#endif // _SINGLETONS_H_