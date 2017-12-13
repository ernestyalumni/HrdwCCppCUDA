# `Cppcls`  

## (Abridged, unfinished) Table of Contents 
- `virtualfunc.cpp` - pointer to inherited class vs. instance of inherited class; how does that change virtual member function?  
- `virtualdes2.cpp` - pointer to inherited class vs. instance of inherited class; how does that change scope, and which destructor?  

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

This vptr is added as a hidden member of this object.  As such, compiler must generate "hidden" code in the **constructors** of each class to initialize a new object's vptr to the address of its class's vtable.  


Whenever there's a virtual function call, vtable is used to resolve to the function address.  
This vptr contains base address of the virtual table in memory.  

Note that virtual tables are class specific, i.e. there's only 1 virtual table for a class, irrespective of number of virtual functions it contains, i.e.  

vtable is same for all objects belonging to the same class, and typically is shared between them.    
  

This virtual table in turn contains base addresses of 1 or more virtual functions of the class.  

At the time when a virtual function is called on an object, the vptr of that object provides the base address of the virtual table for that class in memory.  This table is used to resolve the function call as it contains the addresses of all the virtual functions of that class.  This is how dynamic binding is resolved during a virtual function call, i.e.  

class (inherited or base/parent) cannot, generally, be determined *statically* (i.e. **compile-time**), so compiler can't decide which function to call at that (compile) time.  (Virtual function) call must be dispatched to the right function *dynamically* (i.e. **run-time**).  
  

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


cf. [How are virtual functions and vtable implemented?, stackoverflow](https://stackoverflow.com/questions/99297/how-are-virtual-functions-and-vtable-implemented)

#### Can the vtable be modified or even directly accessed at runtime?  No.  

"Universally, I believe the answer is "no". You could do some memory mangling to find the vtable but you still wouldn't know what the function signature looks like to call it. Anything that you would want to achieve with this ability (that the language supports) should be possible without access to the vtable directly or modifying it at runtime. Also note, the C++ language spec does not specify that vtables are required - however that is how most compilers implement virtual functions."  


#### Does the vtable exist for all objects, or only those that have at least one virtual function?  Only for class with at least 1 virtual function.  

I believe the answer here is "it depends on the implementation" since the spec doesn't require vtables in the first place. However, in practice, I believe all modern compilers only create a vtable if a class has at least 1 virtual function. There is a space overhead associated with the vtable and a time overhead associated with calling a virtual function vs a non-virtual function.


#### Do abstract classes simply have a NULL for the function pointer of at least one entry?

The answer is it is unspecified by the language spec so it depends on the implementation. Calling the pure virtual function results in undefined behavior if it is not defined (which it usually isn't) (ISO/IEC 14882:2003 10.4-2). In practice it does allocate a slot in the vtable for the function but does not assign an address to it. This leaves the vtable incomplete which requires the derived classes to implement the function and complete the vtable. Some implementations do simply place a NULL pointer in the vtable entry; other implementations place a pointer to a dummy method that does something similar to an assertion.

Note that an abstract class can define an implementation for a pure virtual function, but that function can only be called with a qualified-id syntax (ie., fully specifying the class in the method name, similar to calling a base class method from a derived class). This is done to provide an easy to use default implementation, while still requiring that a derived class provide an override.


### Example, from [Virtual method table, wikipedia](https://en.wikipedia.org/wiki/Virtual_method_table), with gdb

```  
#include <iostream>

class B1 {
	public:
		virtual ~B1() {}
		void f0() {}
		virtual void f1() {}
		int int_in_b1;
};  

class B2 {
	public:
		virtual ~B2() {}
		virtual void f2() {}
		int int_in_b2;
};

class D : public B1, public B2 {
	public:
		void d() {}
		void f2() {}  // override B2::f2()
		int int_in_d;
}; 

int main() {
	B2 *b2 = new B2(); 
	D *d   = new D(); 
}  	
```  	

Demonstrating the usage of **info vtbl**, getting information about a virtual method table (vtable) of an object, and **info symbol** to go from address of vtable to actual type of the object.  

```  
// break at line 43 or anywhere before end of main program  
(gdb) b 43 
Breakpoint 1 at 0x4008a3: /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.cpp:43. (3 locations)
(gdb) run
Starting program: /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe 

...

Breakpoint 1, main () at vtable.cpp:49
49	}
(gdb) p b2
$8 = (B2 *) 0x614c20
(gdb) p &b2
$9 = (B2 **) 0x7fffffffde08
(gdb) p d
$10 = (D *) 0x614c40
(gdb) p &d
$11 = (D **) 0x7fffffffde00
(gdb) info vtbl b2
vtable for 'B2' @ 0x400bb0 (subobject @ 0x614c20):
[0]: 0x400952 <B2::~B2()>
[1]: 0x40096a <B2::~B2()>
[2]: 0x400996 <B2::f2()>
(gdb) info vtbl d
vtable for 'D' @ 0x400b58 (subobject @ 0x614c40):
[0]: 0x400a28 <D::~D()>
[1]: 0x400a72 <D::~D()>
[2]: 0x400946 <B1::f1()>
[3]: 0x4009a2 <D::f2()>

vtable for 'B2' @ 0x400b88 (subobject @ 0x614c50):
[0]: 0x400a6c <non-virtual thunk to D::~D()>
[1]: 0x400a9d <non-virtual thunk to D::~D()>
[2]: 0x4009ad <non-virtual thunk to D::f2()>
(gdb) info symbol 0x400bb0
vtable for B2 + 16 in section .rodata of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) info symbol 0x400b58
vtable for D + 16 in section .rodata of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) info symbol 0x400b88
vtable for D + 64 in section .rodata of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe

```  

We can get the, for g++ 6, the following 64-bit memory layout for object `b2`: 
```  

(gdb) p b2->int_in_b2
$12 = 0
(gdb) p &(b2->int_in_b2)
$13 = (int *) 0x614c28
(gdb) info vtbl b2
vtable for 'B2' @ 0x400bb0 (subobject @ 0x614c20):
[0]: 0x400952 <B2::~B2()>
[1]: 0x40096a <B2::~B2()>
[2]: 0x400996 <B2::f2()>

(gdb) p &'vtable for B2'
$15 = (<data variable, no debug info> *) 0x400ba0 <vtable for B2>
(gdb) p &'typeinfo for B2'
$16 = (<data variable, no debug info> *) 0x400c30 <typeinfo for B2>

```  
memory layout for object `d`:  

```  
(gdb) info vtbl d
vtable for 'D' @ 0x400b58 (subobject @ 0x614c40):
[0]: 0x400a28 <D::~D()>
[1]: 0x400a72 <D::~D()>
[2]: 0x400946 <B1::f1()>
[3]: 0x4009a2 <D::f2()>

vtable for 'B2' @ 0x400b88 (subobject @ 0x614c50):
[0]: 0x400a6c <non-virtual thunk to D::~D()>
[1]: 0x400a9d <non-virtual thunk to D::~D()>
[2]: 0x4009ad <non-virtual thunk to D::f2()>
(gdb) p &(d->int_in_b1)
$17 = (int *) 0x614c48
(gdb) p &(d->int_in_b2)
$18 = (int *) 0x614c58
(gdb) p &(d->int_in_d)
$19 = (int *) 0x614c5c

(gdb) p &'vtable for D'
$20 = (<data variable, no debug info> *) 0x400b48 <vtable for D>
(gdb) p &'typeinfo for D'
$21 = (<data variable, no debug info> *) 0x400bf0 <typeinfo for D>

```
Notice that functions `f0`, `d` are not in the vtable.  

Overriding method `f2()` in class `D` is implemented by duplicating vtable of `B2` and replacing pointer to `B2::f2()` with pointer to `D::f2()`, as clearly shown above.  

non-virtual thunk, object vtable entries, used to implement multiple inheritance, does offsetting to the correct derived classobject size to the vtable (by the subtraction or `sub` instruction).  cf. [What is a non-virtual thunk?](https://reverseengineering.stackexchange.com/questions/4543/what-is-a-non-virtual-thunk)

```  
(gdb) info symbol 0x400a6c
non-virtual thunk to D::~D() in section .text of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) info symbol 0x400a9d
non-virtual thunk to D::~D() in section .text of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) info symbol 0x4009ad
non-virtual thunk to D::f2() in section .text of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) disassemble 0x400a6c
Dump of assembler code for function _ZThn16_N1DD1Ev:
   0x0000000000400a6c <+0>:	sub    $0x10,%rdi
   0x0000000000400a70 <+4>:	jmp    0x400a28 <D::~D()>
End of assembler dump.
(gdb) info symbol 0x400a9d
non-virtual thunk to D::~D() in section .text of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) disassemble 0x400a9d
Dump of assembler code for function _ZThn16_N1DD0Ev:
   0x0000000000400a9d <+0>:	sub    $0x10,%rdi
   0x0000000000400aa1 <+4>:	jmp    0x400a72 <D::~D()>
End of assembler dump.
(gdb) info symbol 0x4009ad
non-virtual thunk to D::f2() in section .text of /home/topolo/PropD/HrdwCCppCUDA/Cppcls/vtable.exe
(gdb) disassemble 0x4009ad
Dump of assembler code for function _ZThn16_N1D2f2Ev:
   0x00000000004009ad <+0>:	sub    $0x10,%rdi
   0x00000000004009b1 <+4>:	jmp    0x4009a2 <D::f2()>
End of assembler dump.
```  

As you can clearly see, 2 instructions, `sub`, `jmp`, perform pointer adjustment by subtracting size of the `B2` class.  

cf. [What every C++ programmer should know, The hard part](http://web.archive.org/web/20131210001207/http://thomas-sanchez.net/computer-sciences/2011/08/15/what-every-c-programmer-should-know-the-hard-part/)

*More links*

(I haven't worked this out but this is a very thorough and well written explanation):  

[C++ vtables - Part 1 - Basics (1204 words)](https://shaharmike.com/cpp/vtable-part1/)

## `virtual`  

cf. Ch. 21 Class Hierarchies, **The C++ Programming Language**, Bjarne Stroustrup.  2013.  

