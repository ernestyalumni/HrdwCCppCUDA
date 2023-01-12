use std::process::Command;

fn main()
{
  println!("Running main, printing this with println!");

  let temporary_filename = "./TestData/temporary_test_for_echo.txt";

  // cf. https://www.cyberciti.biz/faq/linux-append-text-to-end-of-file/

  // cf. https://stackoverflow.com/questions/62189939/why-cant-i-execute-some-commands-from-process

  let mut echo_command = Command::new("bash");
  let echo_command = echo_command
    .arg("-c")
    .arg("echo")
    .arg("'Pyka plays great dubstep'")
    .arg(">>")
    .arg(temporary_filename);

  let output = echo_command.output().expect("Failed to execute echo process");
  assert!(output.status.success());

  let output = echo_command.output().expect("Failed to execute echo process");
  assert!(output.status.success());

  let output = echo_command.output().expect("Failed to execute echo process");
  assert!(output.status.success());

  // cf. https://doc.rust-lang.org/std/keyword.await.html
  // Keyword await, suspend execution until result of a Future is ready.
  // .awaiting a future will suspend the current function's execution.

  //await
}
