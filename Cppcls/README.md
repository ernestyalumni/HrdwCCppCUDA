# `Cppcls`  

## vtable, virtual table  


I was given this answer to a question I posed to a 20 year C++ veteran and it was such an important answer (as I did not know a virtual table existed, at all before), that I will copy this, repeat this and explore this extensively:  

"The keyword you're looking for is virtual table: " [How are virtual functions and vtable implemented?, stackoverflow](https://stackoverflow.com/questions/99297/how-are-virtual-functions-and-vtable-implemented)  

Original question, from [Brian R. Bondy](https://stackoverflow.com/users/3153/brian-r-bondy):  

### How are virtual functions and vtable implemented?

We all know what virtual functions are in C++, but how are they implemented at a deep level?

Can the vtable be modified or even directly accessed at runtime?

Does the vtable exist for all classes, or only those that have at least one virtual function?

Do abstract classes simply have a NULL for the function pointer of at least one entry?

Does having a single virtual function slow down the whole class? Or only the call to the function that is virtual? And does the speed get affected if the virtual function is actually overwritten or not, or does this have no effect so long as it is virtual.

Answer from *community wiki*:  

How are virtual functions implemented at a deep level?

#### From ["Virtual Functions in C++"](http://wayback.archive.org/web/20100209040010/http://www.codersource.net/published/view/325/virtual_functions_in.aspx)

Whenever a program has a virtual function declared, a v - table is constructed for the class. The v-table consists of addresses to the virtual functions for classes that contain one or more virtual functions. The object of the class containing the virtual function contains a virtual pointer that points to the base address of the virtual table in memory.  

Whenever there is a virtual function call, the v-table is used to resolve to the function address.  

An object of the class that contains one or more virtual functions contains a virtual pointer called the vptr at the very beginning of the object in the memory. Hence the size of the object in this case increases by the size of the pointer. This vptr contains the base address of the virtual table in memory. Note that virtual tables are class specific, i.e., there is only one virtual table for a class irrespective of the number of virtual functions it contains. This virtual table in turn contains the base addresses of one or more virtual functions of the class.  

At the time when a virtual function is called on an object, the vptr of that object provides the base address of the virtual table for that class in memory. This table is used to resolve the function call as it contains the addresses of all the virtual functions of that class. This is how dynamic binding is resolved during a virtual function call.

cf. ["Virtual Functions in C++"](http://wayback.archive.org/web/20100209040010/http://www.codersource.net/published/view/325/virtual_functions_in.aspx)

##### What is a Virtual Function?

A virtual function is a member function of a class, whose functionality can be over-ridden in its derived classes. It is one that is declared as virtual in the base class using the virtual keyword. The virtual nature is inherited in the subsequent derived classes and the virtual keyword need not be re-stated there. The whole function body can be replaced with a new set of implementation in the derived class. 

##### What is Binding?  

Binding is associating an object or a class with its member.  
If we call a method `fn()` on an object `o` of a class `c`, we say that object `o` is binded with method `fn()`.  

This happens at *compile time* and is known as *static* - or *compile-time* binding.   

Calls to virtual member functions are resolved during *run-time*.  This mechanisms is known as *dynamic-binding.*   

The most prominent reason why a virtual function will be used is to have a different functionality in the derived class.  
The difference between a non-virtual member function and a virtual member function is, the non-virtual member functions are resolved at compile time.  

##### How does a Virtual Function work?  

When a program (code text?) has a virtual function declared, a **v-table** is *constructed* for the class.  

The v-table consists of addresses to virtual functions for classes that contain 1 or more virtual functions.  
The object of the class containing the virtual function *contains a virtual pointer* that points to the base address of the virtual table in memory.  An object of the class that contains 1 or more virtual functions contains a virtual pointer called the **vptr** at the very beginning of the object in the memory.  (Hence size of the object in this case increases by the size of the pointer; "memory/size overhead.")  
Whenever there's a virtual function call, vtable is used to resolve to the function address.  
This vptr contains base address of the virtual table in memory.  

Note that virtual tables are class specific, i.e. there's only 1 virtual table for a class, irrespective of number of virtual functions it contains.  

This virtual table in turn contains base addresses of 1 or more virtual functions of the class.  

At the time when a virtual function is called on an object, the vptr of that object provides the base address of the virtual table for that class in memory.  This table is used to resolve the function call as it contains the addresses of all the virtual functions of that class.  This is how dynamic binding is resolved during a virtual function call.  

```  
#include <iostream>

class base
{
	public:
		virtual void display() 
		{
			std::cout << "\n Base " << std::endl; 
		}
};

class derived : public base 
{
	public:
		void display()
		{
			std::cout << "\n Derived" << std::endl; 
		}
};

void main() {
	base *ptr = new derived(); 
	ptr->display();  
}
```  

`base *ptr` is a pointer of type `base`, but **it points to the `derived` class object**.  
The method `display()` is `virtual` in nature.  Hence, in order to resolve the `virtual` method call, the context of the pointer is considered, i.e., the display method of the derived class is called and not that of the base.  
If method was non-virtual, `display()` method of base class would have been called.  

Remember, calls to virtual member functions are resolved during *run-time*, *dynamic-binding*, and vtable used to resolve function address, and vptr, that the object of the class contains, provides base address of the vtable for that class, in memory.  

##### Virtual Constructors and Destructors  

A constructor cannot be virtual because at the time when constructor is invoked, the vtable wouldn't be available in memory.  Hence, we can't have a virtual constructor.  

A virtual destructor is 1 that's declared as virtual in the base class, and is used to ensure that destructors are called in the proper order.  Remember that destructors are called in reverse order of inheritance.  If a base class pointer points to a derived class object, and we some time later use the delete operator to delete the object, then the derived class destructor is not called.  


## `virtual`  

cf. Ch. 21 Class Hierarchies, **The C++ Programming Language**, Bjarne Stroustrup.  2013.  

