//-------------------------------------------------------------------------------------------------
/// \url https://doc.rust-lang.org/book/ch17-02-trait-objects.html#defining-a-trait-for-common-behavior
/// \details A trait object points to both an instance of a type implementing our specified trait
/// and a table used to look up trait methods on that type at runtime.
/// Create a trait object by specifying some sort of pointer, such as a & reference or a Box<T>
/// smart pointer, then dyn keyword, and then specifying the relevant trait.
/// Use trait objects in place of a generic or concrete type.
//-------------------------------------------------------------------------------------------------

pub struct MutableStruct
{
  pub is_drawn_: bool,
  pub test_message_: String,
  pub test_value_: i32,
}

pub trait Draw
{
  fn draw(&self, input: &mut MutableStruct);
}

pub struct Screen
{
  // A trait object is an object that can contain objects of different types at the same time; the
  // dyn keyword is used when declaring a trait object.
  pub components_: Vec<Box<dyn Draw>>,
}

impl Screen
{
  pub fn run(&self, input: &mut MutableStruct)
  {
    for component in self.components_.iter()
    {
      component.draw(input);
    }
  }
}

//-------------------------------------------------------------------------------------------------
/// Generic type parameter can only be substituted with 1 concrete type at a time, whereas trait
/// objects allow multiple concrete types to fill in for trait object at runtime.
/// This restricts us to a GenericScreen instance that has a list of components all of type
/// StructThatDraw1 or StructThatDraw2.
/// If you'll only ever have homogeneous collections, using generic and trait bounds is preferable
/// because definitions will be monomorphized at compile time to use concrete types.
//-------------------------------------------------------------------------------------------------
pub struct GenericScreen<T: Draw>
{
  pub components_: Vec<T>,
}

impl<T> GenericScreen<T>
where
  T: Draw,
{
  pub fn run(&self, input: &mut MutableStruct)
  {
    for component in self.components_.iter()
    {
      component.draw(input);
    }
  }
}

#[cfg(test)]
mod tests
{
  // The tests module is a regular module that follows the usual visibility rules in Ch. 7,
  // "Paths for Referring to an Item in the Module Tree" of Rust book.
  // Because tests module is an inner module, we need to bring the code under test in the outer
  // module into scope of inner module.
  use super::*;

  // cf. https://www.educative.io/answers/what-is-the-dyn-keyword-in-rust

  struct StructThatDraw1;

  struct StructThatDraw2;

  impl Draw for StructThatDraw1
  {
    fn draw(&self, input: &mut MutableStruct)
    {
      input.is_drawn_ = true;
      input.test_message_ = String::from("draw 1 writes on it");
    }
  }

  impl Draw for StructThatDraw2
  {
    fn draw(&self, input: &mut MutableStruct)
    {
      input.is_drawn_ = true;
      input.test_value_ = 42;
    }
  }

  #[test]
  fn call_draw_on_instances()
  {
    let mut test_input = MutableStruct {
      is_drawn_: false,
      test_message_: String::from(""),
      test_value_: 0};

    // Make this mutable since we need to mutate the Vec<Box<dyn Type>> instance.
    let mut screen = Screen { components_: Vec::new()};
    let instance_1 = StructThatDraw1 {};
    let instance_2 = StructThatDraw2 {};
    screen.components_.push(Box::new(instance_1));
    screen.components_.push(Box::new(instance_2));

    screen.run(&mut test_input);

    assert_eq!(test_input.is_drawn_, true);
    assert_eq!(test_input.test_message_, "draw 1 writes on it");
    assert_eq!(test_input.test_value_, 42);
  }

 #[test]
  fn call_draw_on_generic_type_parameter_struct()
  {
    let mut test_input = MutableStruct {
      is_drawn_: false,
      test_message_: String::from(""),
      test_value_: 0};

    {
      let mut screen = GenericScreen::<StructThatDraw1> { components_: Vec::new()};
      let instance_1 = StructThatDraw1 {};
      let instance_2 = StructThatDraw1 {};
      screen.components_.push(instance_1);
      screen.components_.push(instance_2);

      screen.run(&mut test_input);

      assert_eq!(test_input.is_drawn_, true);
      assert_eq!(test_input.test_message_, "draw 1 writes on it");
      assert_eq!(test_input.test_value_, 0);
    }

    {
      let mut screen = GenericScreen::<StructThatDraw2> { components_: Vec::new()};
      let instance_1 = StructThatDraw2 {};
      let instance_2 = StructThatDraw2 {};
      let instance_3 = StructThatDraw2 {};
      screen.components_.push(instance_1);
      screen.components_.push(instance_2);
      screen.components_.push(instance_3);

      screen.run(&mut test_input);

      assert_eq!(test_input.is_drawn_, true);
      assert_eq!(test_input.test_message_, "draw 1 writes on it");
      assert_eq!(test_input.test_value_, 42);
    }  
  }
}