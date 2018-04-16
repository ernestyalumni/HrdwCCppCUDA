# Templates

cf. pp. 666, Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  


cf. pp. 668, Sec. 23.2 A Simple String Template, Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

```  
template <typename C>
class String {
    public:
        String(); // default constructor
        explicit String(const C*); //
        String(const String&);  // constructor
        String operator=(const String&) // overloading =
        // ...
        C& operator[](int n) { return ptr[n]; } // unchecked element access
        String& operator+=(C c);    // add c at end
        // ...
    private:
        static const int short_max = 15;    // for the short string optimization
        int sz;
        C* ptr; // ptr points to sz Cs
};
```  

Mathematicians will recognize `template<typename C>` as $\forall \, C$, "for all C" or more specifically, $\forall \, \text{ types } C$, "for all types C" or even $\forall \, C, \text{ s.t. } C \text{ is a type }$, "for all C, such that C is a type".  

If you think along these lines, note C++ lacks a fully general mechanism for specifying required properties of template parameter `C`, i.e. C++ doesn't offer a direct way to say what kind of type a template argument `C` is supposed to be (Sec. 24.3).  



cf. pp. 669, Sec. 23.2.1 "Defining a Template", Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  

Use of a template doesn't imply any run-time mechanisms beyond what's used for an equivalent "handwritten" class.  In fact, using template can lead to a decrease of code generated because code for a member function of a class template is only generated if that member is used (Sec. 26.2.1).  

Template is a specification of how to generate something given suitable template arguments; language mechanisms for doing that generation (instantiation (26.2) and specialization (25.3)) don't care much whether a class or a function is generated.  

Consider *class template* and *template class* interchangeable, and *function template* and *template function* interchangeable.  



## Function Templates 

cf. pp. 684, Sec. 23.5 *Function Templates*, Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*.  


