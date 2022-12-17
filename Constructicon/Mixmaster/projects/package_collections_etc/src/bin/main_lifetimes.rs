use package_collections_etc::generics_traits_lifetimes::first_word;
use package_collections_etc::generics_traits_lifetimes::longest;
use package_collections_etc::generics_traits_lifetimes::ImportantExcerpt;

fn main()
{
  let lang = String::from("lang");
  let langste = String::from("langste");

  let doorverwijzing = longest(&lang, &langste);

  println!("doorverwijzing string: {}", doorverwijzing);

  /*
  let referenza = &str();

  {
    let lungo = String::from("lungo");
    let ilpiulungo = String::from("ilpiulungo");   
    referenza = longest(&lungo, &ilpiulungo);
  }
  */

  // cf. https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html

  // string1 is valid until end of the outer scope.
  let string1 = String::from("long string is long");

  {
    // string2 is valid until end of inner scope.
    let string2 = String::from("xyz");
    // result references something that's valid until end of the inner scope.
    let result = longest(string1.as_str(), string2.as_str());
    println!("The longest string is {}", result);
  }

  let novel = String::from("Call me Ishmael. Some years ago...");
  let first_sentence = novel.split('.').next().expect("Could not find a '.'");

  // novel doesn't go out of scope until after ImportantExcerpt goes out of scope, so reference in
  // ImportantExcerpt instance is valid.
  let i = ImportantExcerpt {
    part_: first_sentence,
  };

  println!("Quote: {}", i.part_);

  println!("First Word: {}", first_word(novel.as_str()));
}