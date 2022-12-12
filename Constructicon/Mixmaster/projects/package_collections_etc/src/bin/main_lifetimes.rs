use package_collections_etc::generics_traits_lifetimes::longest;

fn main()
{
  let lang = String::from("lang");
  let langste = String::from("langste");

  let doorverwijzing = longest(&lang, &langste);

  println!("doorverwijzing string: {}", doorverwijzing);
}