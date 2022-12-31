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
    let current_directory = env::current_dir().expect(
      "Failed to get current directory");

    let current_directory = current_directory.join("src");

    assert!(current_directory.exists()); 

    let mut entries = Vec::new();

    // https://doc.rust-lang.org/std/path/struct.PathBuf.html

    for entry in current_directory.read_dir().expect("read_dir call failed")
    {
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

  #[test]
  fn command_runs_system_commands()
  {
    {
      let output = Command::new("ls").arg("-ls").output().expect("Failed to execute process ls");
      assert!(output.status.success());
    }
    {
      let output = Command::new("echo").arg("'dark'").output().expect("Failed to execute echo");
      assert!(output.status.success());
    }
  }
}
