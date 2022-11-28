#/bin/bash

# Make this an executable with `chmod` or run with `bash Setup.sh`

# cf. https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script

echo "Do you wish to install rust?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; break;;
        No ) exit;;
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
