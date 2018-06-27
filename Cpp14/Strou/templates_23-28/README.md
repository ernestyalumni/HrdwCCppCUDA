# Templates

## Templates

cf. pp. 665, Ch. 23 **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

### A Simple String Template, `./String/`

Consider a string of characters. A string is a class that holds characters and provides operations such as subscripting, concatenation, and comparison that we usually associate with the notion of a "string."

We want to represent the notion of "string" with minimal dependence on a specific kind of character. The definition of a string relies on the fact that a character can be copied, and little else (Sec. 24.3). Thus, we can make a more general string type by taking the string of `char` from Sec. 19.3 and making the character type a parameter.

`template<typename C>` prefix specifies template is being declared and that a type argument `C` will be used in declaration. After its introduction, `C` used exactly like other type names. 
  `C` is a *type* name, need not be name of a *class* `template<class C>`. 

Mathematicians will recognize `template<typename C>` as a variant of "for all C" or "for all types C" or even "for all C, such that C is a type." 
  If you think along those lines, you'll note that C++ lacks a fully general mechanism for specifying the required properties of a template parameter `C`, i.e. we **can't** say "for all C, such that ..." where the "..." is a set of requirements for `C`. 
i.e. C++ doesn't offer a direct way to say what kind of type a template argument `C` is supposed to be (Sec. 24.3).

The standard library provides template class `basic_string` that's similar to templatized `String` (Sec. 19.3, 36.3). In standard library, `string` is a synonym for `basic_string<char>` (Sec. 36.3).

### Defining a Template

cf. pp. 669, Ch. 23.2.1 Defining a Template. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. Ch. 23.

A class generated from a class template is a perfectly ordinary class. Thus, use of a template doesn't imply any run-time mechanisms beyond what's used for an equivalent "handwritten" class. In fact, using template can lead to decrease of code generated because code for a member function of a class template is only generated if that member is used. (Sec. 26.2.1).

A template is a specification of how to generate something given suitable template arguments; the language mechanisms for doing that generation (instantiation (Sec. 26.2) and specialization (Sec. 25.3)) don't care much whether a class or function is generated. 

Members of a template class are themselves templates parametrized by parameters of their template class. When such a member is defined outside its class, it must explicitly be declared a template. e.g. 

```
template <typename C>
String<C>::String(): // String<C>'s constructor
  sz_{0}, ptr_{ch}
{
  ch[0] = {}; // terminating 0 of the appropriate character type
}

```

### Template Instantiation

cf. pp. 671, Ch. 23.2.2 Template Instantiation. **Templates** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. Ch. 23.

*template instantiation* (Sec. 26.2) - process of generating a class or a function from a template plus a template argument list.

In general, it's the implementation's job - *not* the programmer's - to ensure that specializations of a template are generated for each template argument list used.



