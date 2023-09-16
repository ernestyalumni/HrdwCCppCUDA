use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

//------------------------------------------------------------------------------
/// \details Use Newtype pattern, creating a new struct that wraps original
/// type.
//------------------------------------------------------------------------------
struct Set<T>(HashSet<T>);

impl<T> Set<T>
{
	pub fn new() ->Self
	{
		Set(HashSet::new())
	}
}

//------------------------------------------------------------------------------
/// \details Use Deref Trait for Transparent Access
//------------------------------------------------------------------------------
impl<T> Deref for Set<T>
{
	// https://web.mit.edu/rust-lang_v1.25/arch/amd64_ubuntu1404/share/doc/rust/html/book/second-edition/ch15-02-deref.html
	// type Target = T; syntax defines associated type for Deref trait to use.
	type Target = HashSet<T>;

	fn deref(&self) -> &Self::Target
	{
		// https://doc.rust-lang.org/std/keyword.self.html
		// Receiver of the current module, referencing current module, refers to
		// Set<T>
		&self.0
	}	
}

impl<T> DerefMut for Set<T>
{
	fn deref_mut(&mut self) -> &mut Self::Target
	{
		&mut self.0
	}
}

//------------------------------------------------------------------------------
/// Unit tests
//------------------------------------------------------------------------------

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn constructs_with_type()
  {
  	let mut set = Set::<String>::new();

  	assert!(true);
  }

  #[test]
  fn inserts()
  {
  	let mut set = Set::<String>::new();

  	set.insert("A Dance With Dragons".to_string());
  	set.insert("To Kill a Mockingbird".to_string());
  	set.insert("The Odyssey".to_string());
  	set.insert("The Great Gatsby".to_string());

  	assert!(set.contains(&String::from("A Dance With Dragons")));
  	assert!(set.contains(&String::from("To Kill a Mockingbird")));
  	assert!(set.contains(&String::from("The Odyssey")));
  	assert!(set.contains(&String::from("The Great Gatsby")));

  	assert!(!set.contains(&String::from("Childhood's End")));
  }
}