pub mod defining_structs;

fn main() {
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

}
