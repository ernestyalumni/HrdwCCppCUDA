#/bin/bash

# Make this an executable with `chmod` or run with `bash Setup.sh`
# cf. https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script

# cf. https://github.com/rust-lang/rustup/issues/686
# Once you're done, you want to do
# source ~/.profile

echo "Do you wish to install rust?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; break;;
        No ) break;
#        No ) exit;;
    # cf. https://unix.stackexchange.com/questions/256149/what-does-esac-mean-at-the-end-of-a-bash-case-statement-is-it-required
    # esac is backwards for case and is proper way to close case.
    esac
done

echo "Do you wish to update rust?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) rustup update; break;;
        No ) exit;;
    esac
done

# Run the following to access the Rust book offline
# rustup docs --book 