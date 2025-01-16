#!/bin/bash

echo "WARNING"
echo "This script will:"
echo "* DELETE the contents of ~/neuromorphic"
echo "* DELETE data stored in turtwig/results and turtwig/out"
echo "* DELETE your pyenv installation"
echo "* DELETE files related to libffi and autoconf, even if they're being used by other apps"
echo "* DELETE this script and the deploy.sh install script (you'll need to redownload it)"
echo
echo "Running this script may result in data loss!"
echo "If you understand and want to continue, tap 'y' on your keyboard."
echo "If you need more granular cleanup, consider reading this script"
echo "and manually deleting the stuff you want to remove."

# https://stackoverflow.com/a/1885534
read -p "Press 'y' to continue: " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

echo "To re-install, you can do: "
echo "git clone https://gitlab.orc.gmu.edu/kzhu4/neuromorphic_experiments.git ~/neuromorphic/turtwig"
echo "~/neuromorphic/turtwig/scripts/hopper/deploy.sh"
echo

if [ ! -z ${VIRTUAL_ENV+x} ];  # deactivate if we're in a venv
then
    echo "Deactivating virtual environment"
    deactivate
fi

BASHRC=$HOME/.bashrc

function removefrom_bashrc {
    LINE=$1
    grep -v "$LINE" $BASHRC > temp && mv temp $BASHRC
    return
}

echo "Removing pyenv from ~/.bashrc"
removefrom_bashrc 'export PYENV_ROOT="$HOME/.pyenv"'
removefrom_bashrc 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
removefrom_bashrc 'eval "$(pyenv init -)"'

set +x  # print commands
rm -rf ~/privatemodules/autoconf-2.72
rm -rf ~/privatemodules/libffi-3.4.6

rm -rf ~/.pyenv

rm -rf ~/.local/autoconf-2.72*  # also removes duplicate downloads
rm -rf ~/.local/packages/autoconf-2.72
rm -rf ~/.local/libffi-3.4.6*  # also removes duplicate downloads
rm -rf ~/.local/packages/libffi-3.4.6

rm -rf ~/neuromorphic
set -x

echo "So long, and thanks for all the fish"