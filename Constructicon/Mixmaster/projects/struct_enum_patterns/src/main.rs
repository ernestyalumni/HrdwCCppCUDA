pub mod defining_structs;

fn main()
{
  // Create an instance of the struct by specifying concrete values for each fields.
  let user1 = defining_structs::User {
      email: String::from("someone@example.com"),
      username: String::from("someusername123"),
      active: true,
      sign_in_count: 1,
  };

  println!("user1: {} ", user1.email);

  println!("Hello, world!");

  let user2 = defining_structs::build_user(
      String::from("some2@example.com"),
      String::from("created"));

  println!("dear god: {}, {}", user2.username, user2.sign_in_count);

  let user3 = defining_structs::build_user2(
      String::from("some3@example.com"),
      String::from("created3"));

  println!("dear god: {}", user3.username);

  let new_user3 = defining_structs::User {
      email: String::from("another3@example.com"),
      // ..user3 must come last to specify that any remaining fields should get their values from
      // the corresponding fields in user3.
      ..user3
  };

  println!("gift subs: {} {}", new_user3.email, new_user3.username);

  let black = defining_structs::Color(0, 1, 2);

  let origin = defining_structs::Point(2, 1, 0);

  println!("black {} {} ", black.0, black.2);
  println!("origin {} {} ", origin.1, origin.2);

  let rect1 = defining_structs::Rectangle {
    width: 30,
    height: 50,
  };

  println!{" with no hash {:?}", rect1};

  // Useful format for large structs.
  println!(" with hash {:#?}", rect1);

  // dbg! macro prints to standard error console.
  dbg!(&rect1);

  println!("The area of the rectangle is {} square pixels.", rect1.area());

  let rect2 = defining_structs::Rectangle
  {
    width: 10,
    height: 40,
  };

  let rect3 = defining_structs::Rectangle
  {
    width: 60,
    height: 45,
  };

  println!("Can rect1 hold rect2 {}", rect1.can_hold(&rect2));
  println!("Can rect1 hold rect3 {}", rect1.can_hold(&rect3));

  let rect4 = defining_structs::Rectangle::square(3);

  println!("Area of square is {}", rect4.area());
}
