use package_collections_etc::generics_traits_lifetimes::{
  Summary,
  Tweet,
  largest};

fn main()
{
  // cf. https://doc.rust-lang.org/stable/std/primitive.array.html
  // [x; N] which produces an array with N copies of x.
  let mut example_list: [i32; 4] = [0; 4];
  example_list[0] = 69;
  example_list[3] = 42;

  largest(&example_list);

  let tweet = Tweet
  {
    username: String::from("horse_ebooks"),
    content: String::from(
      "of course, as you probably already know, people",
      ),
    reply: false,
    retweet: false,
  };

  println!("1 new tweet: {}", tweet.summarize());
}