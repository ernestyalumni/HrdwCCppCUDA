pub mod defining_enums;
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

  //-----------------------------------------------------------------------------------------------
  // Enums and Pattern Matching
  // \ref https://doc.rust-lang.org/book/ch06-00-enums.html
  //-----------------------------------------------------------------------------------------------

  let four = defining_enums::IpAddrKind::V4;
  let six = defining_enums::IpAddrKind::V6;

  println!("four : {:?}", four);
  println!("six : {:?}", six);


  let home = defining_enums::IpAddr::V4(String::from("127.0.0.1"));
  println!("home: {:?}", home);

  let _q = defining_enums::Message::Quit;
  let _m = defining_enums::Message::Write(String::from("hello"));
  let _c1 = defining_enums::Message::ChangeColor(3, 1, 2);

  // Rust can infer these types because we've specified a value inside the Some variant.
  let _some_number = Some(5);
  let _some_char = Some('e');

  // Rust requires us to annotate the overall Option type: the compiler can't infer the type that
  // the corresponding Some variant will hold by looking only at a Non.
  let _absent_number: Option<i32> = None;

  // Get the inner state value out of Coin enum variant for Quarter.
  println!("value: {}",
    defining_enums::value_in_cents(defining_enums::Coin::Quarter(
      defining_enums::UsState::Alaska)));

  //-----------------------------------------------------------------------------------------------
  // Matching with Option<T>
  //-----------------------------------------------------------------------------------------------

  let five = Some(5);
  let six = defining_enums::plus_one(five);
  let none = defining_enums::plus_one(None);
  println!("six: {:#?}", six);
  println!("none: {:?}", none);

  // Demonstrates that particular values don't get caught.
  println!("zero: {:#?}", defining_enums::plus_one(Some(0)));

  println!("42: {:#?}", defining_enums::plus_one(Some(42)));

  //-----------------------------------------------------------------------------------------------
  // Concise Control Flow with if let.
  // \ref https://doc.rust-lang.org/book/ch06-03-if-let.html#concise-control-flow-with-if-let
  //-----------------------------------------------------------------------------------------------

  // Case: we only want to match on a single value. We don't want to do anything with None value.
  let config_max = Some(3u8);
  match config_max
  {
    Some(max) => println!("The maximum is configured to be {}", max),
    _ => (),
  }

  // Instead, write in a shorter way using if let.
  let config_max = Some(3u8);
  if let Some(max) = config_max
  {
    println!("The maximum is configured to be {}", max);
  }

  // We can include an else with an if let.

  let coin = defining_enums::Coin::Penny;

  let mut count = 0;
  if let defining_enums::Coin::Quarter(state) = coin
  {
    println!("State quarter from {:?}!", state);
  }
  else {
    count += 1;
  }

  println!("count result: {}", count);
}
