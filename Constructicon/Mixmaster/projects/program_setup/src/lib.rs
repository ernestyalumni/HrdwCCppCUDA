use std::{env};
use std::ffi::OsStr;
use std::process::Command;

#[cfg(test)]
mod tests {
  
  use super::*;

  #[test]
  fn get_current_directory()
  {
    // Cannot use ? operator in a function that returns ()
    //let current_directory = env::current_dir()?;
    let current_directory = match env::current_dir()
    {
      Ok(current_directory_obtained) =>
      {
        current_directory_obtained
      },
      Err(current_directory_error) =>
      {
        //eprintln!("Failed to get current_directory, {current_directory_error}");
        panic!("Failed to get current_directory, {current_directory_error}");
      },
    };

    // e.g. "/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Mixmaster/projects/program_setup"
    //assert_eq!(current_directory.to_str(), Some("/"));

    assert!(current_directory.exists());
  }

  #[test]
  fn join_adds_to_current_directory()
  {
    let current_directory = env::current_dir().expect(
      "Failed to get current directory");

    let current_directory = current_directory.join("src");

    assert!(current_directory.exists()); 
  }

  #[test]
  fn add_all_directory_entries_to_vec()
  {
    // current_dir() -> Result<PathBuf>
    let current_directory = env::current_dir().expect(
      "Failed to get current directory");

    let current_directory = current_directory.join("src");

    assert!(current_directory.exists()); 

    let mut entries = Vec::new();

    // https://doc.rust-lang.org/std/path/struct.PathBuf.html
    // read_dir() -> Result<ReadDir> 
    for entry in current_directory.read_dir().expect("read_dir call failed")
    {
      // entry is of type DirEntry
      if let Ok(entry) = entry
      {
        entries.push(entry);
      }
    }

    let mut all_entry_final_components = Vec::new();

    // Wrong attempt.
    // let mut temporary_entry : std::fs::DirEntry;

    for entry in entries
    {
      // Wrong attempts.
      /*
      let mut temporary_path = std::path::PathBuf::new();
      temporary_path = entry.path();
      let file_name_entry = temporary_path.file_name();
      let file_name_entry = file_name_entry.unwrap();
      */

      let all_entry_final_component = String::from(
        entry.path().file_name().unwrap().to_str().unwrap());

      // https://doc.rust-lang.org/std/fs/struct.DirEntry.html
      // pub fn path(&self) -> PathBuf
      // cf. https://web.mit.edu/rust-lang_v1.25/arch/amd64_ubuntu1404/share/doc/rust/html/std/path/struct.PathBuf.html
      all_entry_final_components.push(all_entry_final_component);
    }

    //let expected_entry_final_component = OsStr::new("lib.rs");
    let expected_entry_final_component = String::from("lib.rs");

    assert!(all_entry_final_components.contains(&expected_entry_final_component));
  }

  // cf. https://blog.logrocket.com/command-line-argument-parsing-rust-using-clap/
  #[test]
  fn add_all_directory_file_entries_to_vec_of_tuples()
  {
    // error: ? can only be used in a function that returns Result or Option. 
    // let current_directory = env::current_dir()?;
    
    let current_directory = env::current_dir().expect("Failed to get current directory");
    let current_directory = current_directory.join("src");
    assert!(current_directory.exists());

    let mut results = Vec::<(String, std::fs::DirEntry)>::new();

    for entry in current_directory.read_dir().expect("read dir call failed")
    {
      if let Ok(entry) = entry
      {
        // Instead, use is_file directly rather than this:
        // if !entry.file_type().expect("file_type call failed").is_dir()
        if entry.file_type().expect("file_type call failed").is_file()
        {
          let entry_final_component = String::from(
            entry.path().file_name().unwrap().to_str().unwrap());
          results.push((entry_final_component, entry));
        }
      }
    }

    // error: cannot move out of 'k.0' which is behind a shared reference
    // results.sort_by_key(|k| k.0);
    results.sort_by(|a, b| a.0.cmp(&b.0));
  }

  #[test]
  fn command_runs_system_commands()
  {
    // Command::output() is blocking.
    {
      let output = Command::new("ls").arg("-ls").output().expect("Failed to execute process ls");
      assert!(output.status.success());

    }
    {
      let output = Command::new("echo").arg("'dark'").output().expect("Failed to execute echo");
      assert!(output.status.success());
    }
  }

  #[test]
  #[should_panic]
  fn command_panics_when_sudo_is_combined_with_command()
  {
    let output = Command::new("sudo ls").arg("-ls").output().expect(
      "Failed to execute process ls");
    assert!(output.status.success());
  }

  // This test passes when the user account has sudo privileges.
  #[test]
  fn command_runs_sudo_as_a_command()
  {
    /*
    {
      let output = Command::new("sudo").arg("ls").arg("-ls").output().expect(
        "Failed to execute process ls");
      assert!(output.status.success());
    }
    */
  }
}
