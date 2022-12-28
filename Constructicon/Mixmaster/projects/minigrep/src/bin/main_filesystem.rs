fn main()
{
  // return type std::io::Result<PathBuf> for std::env::current_dir()

  /*
  if let Ok(current_working_directory_value) = std::env::current_dir()
  {
    println!("current working directory value: {:?}", current_working_directory_value);
  }
  else
  {
    println!("error occurred: {}", error.kind);
  }
  */

  // cf. https://doc.rust-lang.org/rust-by-example/std/result.html
  // of type PathBuf
  let mut current_working_directory_result = match std::env::current_dir()
  {
    Err(error) => panic!("{}:?", error),
    Ok(value) => value,
  };
  // Doesn't work without panic because of type mismatch with enum ErrorKind from error.kind().
  // Current directory is where it is run from, so if you run cargo run --bin main_filesystem
  println!("current dir: {}", current_working_directory_result.display());

  /*
  let current_dir_result = match env::current_dir()
  {
    Ok(path) => path,
    Err(error) => 
  }
  */

  //-----------------------------------------------------------------------------------------------
  // \brief Reading a File
  //-----------------------------------------------------------------------------------------------

  // "/Data/poem.txt", since it has a root, it replaces everything except for prefix (if any) of
  // self.
  current_working_directory_result.push("Data/poem.txt");

  println!("current dir: {}", current_working_directory_result.display());

  // https://doc.rust-lang.org/std/fs/fn.read_to_string.html
  // pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String>
  let contents = std::fs::read_to_string(current_working_directory_result)
    .expect("Should have been able to read the file");

  println!("With text:\n{contents}");
}