use package_collections_etc::SpreadsheetCell;

use std::collections::HashMap;

fn main()
{
  let v: Vec<i32> = Vec::new();
  println!("empty v is empty: {}", v.is_empty());

  // Rust provides vec! macro which will create a new vector that holds the values you give it.
  let v = vec![1, 2, 3];

  println!("2nd vector element: {}", v[1]);

  // cf. https://doc.rust-lang.org/book/ch08-01-vectors.html#updating-a-vector
  // Updating a Vector

  let mut v = Vec::new();
  v.push(5);
  v.push(6);
  v.push(7);
  v.push(8);

  println!("4th vector element: {}", v[3]);

  // cf. https://doc.rust-lang.org/book/ch08-01-vectors.html#reading-elements-of-vectors
  // Reading Elements of Vectors

  let v = vec![1, 2, 3, 4, 5];

  // Using & and [] gives us a reference to the element at the index value.
  let third: &i32 = &v[2];
  println!("the third element is {}", third);

  // When we use get method with index passed as argument, we get an Option<&T> that we can use
  // with match.
  let third: Option<&i32> = v.get(2);
  match third
  {
    Some(third) => println!("The third element is {}", third),
    None => println!("there is no third element."),
  }

  // thread panics.
  // let does_not_exist = &v[100];
  let does_not_exist = v.get(100);
  println!("does_not_exist: {:?}", does_not_exist);

  let mut v = vec![1, 2, 3, 4, 5];
  let first = &v[0];

  // mutable borrowing. error due to vector might require allocating new memory and copying the old
  // elements to the new space.
  // v.push(6);
  println!("The first element is: {}", first);
  let first = &mut v[0];
  *first = 42;
  println!("The first element is: {}", first);

  // Iterating over the Values in a Vector
  let v = vec![100, 32, 57];
  for i in &v
  {
    println!("{}", i);
  }

  let mut v = vec![100, 32, 57];
  for i in &mut v
  {
    *i += 50;
  }

  let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("blue")),
    SpreadsheetCell::Float(10.12),
  ];

  if let SpreadsheetCell::Float(row_value) = row[2]
  {
    println!("The row value is: {:}", row_value);
  }

  //-----------------------------------------------------------------------------------------------
  // Creating a New String
  //-----------------------------------------------------------------------------------------------

  let mut s = String::new();
  println!("empty string s: {s}");
  s.push('x');

  // Start string with some initial data, to_string method on any type that implements Display
  // trait.
  let data = "initial contensts";
  let s = data.to_string();
  println!("string s: {s}");

  // The method also works on a literal directly:
  let s = "initial contents".to_string();
  println!("string s: {s}");

  let s = String::from("initial contents");
  println!("string s: {s}");

  //-----------------------------------------------------------------------------------------------
  // Updating a String
  //-----------------------------------------------------------------------------------------------

  let mut s = String::from("foo");
  // push_ptr takes a string slice because we don't necessarily want to take ownership of the
  // parameter.
  s.push_str("bar");

  let mut s1 = String::from("foo");
  let s2 = "bar";
  // If push_str method took ownership of s2, we wouldn't be able to print its value on the last
  // line.
  s1.push_str(s2);
  println!("s2 is {}", s2);

  let mut s = String::from("lo");
  s.push('l');
  println!("s: {s}");

  //-----------------------------------------------------------------------------------------------
  // Concatenation with the + Operator or the format! Macro
  //-----------------------------------------------------------------------------------------------

  let s1 = String::from("Hello, ");
  let s2 = String::from("world!");
  // Note s1 has been moved here and can no longer be used.
  let s3 = s1 + &s2;
  println!("s3: {s3}");
  println!("s2: {s2}");
  // error: borrow of moved value.
  //println!("s1: {s1}");

  // For more complicated string combining, instead use format! macro.

  let s1 = String::from("tic");
  let s2 = String::from("tac");
  let s3 = String::from("toe");

  // format! macro uses references so that this call doesn't take ownership of any of its
  // parameters.
  let s = format!("{}-{}-{}", s1, s2, s3);
  println!("s: {s}");
  println!("s2: {s2}");

  for c in "hello_world".chars()
  {
    println!("{}", c);
  }

  for b in "hello_world".bytes()
  {
    println!("{}", b);
  }

  //-----------------------------------------------------------------------------------------------
  // cf. https://doc.rust-lang.org/book/ch08-03-hash-maps.html
  // Storing Keys with Associated Values in Hash Maps
  //-----------------------------------------------------------------------------------------------

  let mut scores = HashMap::new();
  scores.insert(String::from("Blue"), 10);
  scores.insert(String::from("Yellow"), 50);

  // Accessing Values in a Hash Map

  let team_name = String::from("Blue");
  // get method returns an Option<&V>;
  // call copied to get an Option<i32> rather than an Option<&i32>,
  // then unwrap_or to set score to zero if scores doesn't have an entry for the key.
  let score = scores.get(&team_name).copied().unwrap_or(0);
  println!("score: {score}");

  if let Some(value) = scores.get(&String::from("Black"))
  {
    println!("found a value: {value}");
  }
  else
  {
    println!("found None!");
  }

  for (key, value) in &scores
  {
    println!("{}: {}", key, value);
  }

  //-----------------------------------------------------------------------------------------------
  // cf. https://doc.rust-lang.org/book/ch08-03-hash-maps.html#hash-maps-and-ownership
  // Hash Maps and Ownership
  // For types that implement the Copy trait, like i32, values are copied into the hash map.
  // For owned values like String, values are moved and hash map will be owner of those values.
  // If we insert references to values into hash map, the values won't be moved into the hash map.
  // The values that the references point to must be valid for at least as long as the hash map is
  // valid.
  //-----------------------------------------------------------------------------------------------
  let field_name = String::from("Favorite color");
  let field_value = String::from("Blue");
  let field_name2 = String::from("Next");
  let field_value2 = String::from("Black");

  let mut map = HashMap::new();
  map.insert(&field_name, &field_value);
  map.insert(&field_name2, &field_value2);

  for (key, value) in &map
  {
    println!("{}: {}", key, value);
  }

  //-----------------------------------------------------------------------------------------------


}