"""
@brief Demonstrate build-in function super() with respect to inheritance.

@url https://realpython.com/python-super/

@details

super([type[,object-or-type]])

Return proxy object that delegates method calls to a parent or sibling class of
*type*. Useful for accessing inherited methods that have been overridden in a
class.

The *object-or-type* determines method resolution order to be searched. Search
starts from class right after the *type*.

For example, if __mro__ (method resolution order) of *object-or-type* is
D -> B -> C -> A -> object and value of *type* is B, then super() searches
C -> A -> object.

__mro__ attribute of *object-or-type* lists method resolution search order used
by both getattr() and super(). 

If 2nd. argument is omitted, super object returned is unbound. If 2nd. argument
is an object, `isinstance(obj, type)` must be true.

cf. https://docs.python.org/3/library/functions.html#super
"""

"""
super() in Single Inheritance
@url https://realpython.com/python-super/#super-in-single-inheritance
"""

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width


class SquareCopy:
    def __init__(self, length):
        self.length = length

    def area(self):
        return self.length * self.length

    def perimeter(self):
        return 4 * self.length


"""
Instead, by using inheritance, you can reduce amount of code you write while
simultaneously reflect real-world relationship between rectangles and squares:
"""


# Here we declare Square class inherits from Rectangle class
class Square(Rectangle):
    """
    @details Even though Square class doesn't explicitly implement it,
    the call to .area() will use .area() method in superclass and print 16. The
    Square class inherited .area() from Rectangle class.
    """
    def __init__(self, length):
        # In Python 3 parameterless super() call equivalent to
        # super(Square, self).__init__(length, length)
        super().__init__(length, length)


"""
What Can super() Do for you in single inheritance?

Allows you to call methods of superclass in subclass; primary use case is to
extend functionality of inherited method.
"""

class Cube(Square):
    """
    @details Notice that Cube class doesn't have .__init__(), because Cube
    inherits from SquareInherited and .__init__() doesn't really do anything
    differently for Cube than it already does for SquareInherited, and so the
    __init__() of the superclass SquareInherited will be called automatically.
    """
    def surface_area(self):
        """
        @details Rather than reimplementing area calculation, use super.
        """
        # face_area = super(Square, self).area()
        face_area = super().area()
        return face_area * 6


    def volume(self):
        """
        @details Rather than reimplementing area calculation, use super.
        """
        # face_area = super(Square, self).area()
        face_area = super().area()
        return face_area * self.length


"""
super() in Multiple Inheritance

@url https://realpython.com/python-super/#super-in-multiple-inheritance

Python supports multiple inheritance, in which a
subclass can inherit from multiple superclasses that don't necessarily
inherit from each other (aka sibling classes)

TODO: Consider using a class member to keep track of when a class gets
initialized. cf. https://stackoverflow.com/questions/40457599/how-to-check-if-an-object-has-been-initialized-in-python/40458701
"""

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    #def area(self):
    def tri_area(self):
        return 0.5 * self.base * self.height


class RightPyramid(Square, Triangle):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height

        # super().__init__(self.base) has TypeError __init__() missing 1
        # required positional argument: 'height'
        # Square is a subtype of the type Rectangle.
        # This calls all constructors up to (and including) Square.
        # cf. https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        super(RightPyramid, self).__init__(self.base)
        #uper().__init__(self.base)

        # This calls all ctors after Rectangle, up to Triangle.
        # cf. https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        # It's a mistake to do
        # super(Square, self).__init__(...)
        # because of the __mro__ (method resolution order) because the type
        # that comes after Square is Rectangle.
        super(Rectangle, self).__init__(self.base, self.slant_height)

    def area(self):
        #base_area = super(Square, self).area()
        base_area = super().area()

        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area

    def area_2(self):
        #base_area = super(Square, self).area()

        base_area = super().area()

        #triangle_area = super().tri_area()
        #triangle_area = super(Square, self).tri_area()
        triangle_area = self.tri_area()

        return triangle_area * 4 + base_area
        #return


"""
mixin - Multiple Inheritance alternative

Instead of defining an "is-a" relationship, define an "includes-a"
relationship. With a mix-in, you can write behavior that can be directly
included in any number of other classes.
"""

class VolumeMixin:
    """
    @details This mixin can be used the same way in any other class that has an
    area defined for it and for which formula area * height returns the correct
    volume.
    """
    def volume(self):
        return self.area() * self.height

class CubeMixedIn(VolumeMixin, Square):
    def __init__(self, length):
        super().__init__(length)
        self.height = length

    def face_area(self):
        return super().area()

    def surface_area(self):
        return super().area() * 6