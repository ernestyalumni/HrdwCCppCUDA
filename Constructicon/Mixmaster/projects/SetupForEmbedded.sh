#/bin/sh

# List all supported targets.
rustc --print target-list

# Look for arm-linux-gnu* for type of compiler
ls -al /usr/bin/
ls -al /usr/bin/ | grep -A 2 -B 2 arm-linux-*

rustup target add armv7-unknown-linux-gnueabihf

cargo build --target=armv7-unknown-linux-gnueabihf --release