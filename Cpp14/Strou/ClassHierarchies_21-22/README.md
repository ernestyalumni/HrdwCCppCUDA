# Derived Classes

cf. pp. 613, Ch. 21 **Class Hierarchies** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

## Implementation Inheritance

cf. pp. 614, Sec. 21.2.1. Implementation Inheritance. Ch. 21 **Class Hierarchies** by Bjarne Stroustrup, **The C++ Programming Language**, *4th Ed.*. 

Class `IValBox` defined basic interface to all `IValBox`es, and specified default implementation that more specific kinds of `IValBox` can override with their own versions.

```
# "Integer Value Input Box"
class IValBox
{
  protected:

    int val;
    int low, high;
    bool changed {false}; // changed by user using set_value()

  public:

    IValBox(int ll, int hh) : val {ll}, low {ll}, high {hh}
    {}

    virtual int get_value()
    {
      changed = false;
      return val;
    } // for application

    virtual void set_value(int i)
    {
      changed = true;
      val = i;
    } // for user

    virtual void reset_value(int i)
    {
      changed = false;
      val = i;
    } // for application

    virtual void prompt() 
    {}

    virtual bool was_changed() const
    {
      return changed;
    }

    virtual ~IValBox() {};
}
```
