#!/bin/bash

echo "WARNING"
echo "This script will:"
echo "* DELETE the contents of ~/neuromorphic"
echo "* DELETE data stored in turtwig/results and turtwig/out"
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

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd $parent_path

ORIGINURL=$(git remote get-url origin)
GOT_ORIGINURL=$?
if [ $GOT_ORIGINURL -ne 0 ]; then
    echo "Could not get origin url from git remote."
    echo "To re-download, fork https://github.com/GMU-ASRC/neuroswarm "
    echo "Clone your fork via ssh to ~/neuromorphic/neuroswarm ,"
    echo "Then run ~/neuromorphic/neuroswarm/scripts/hopper/deploy.sh"
else
    echo
    echo "To re-install, you can do: "
    echo
    echo "git clone $ORIGINURL ~/neuromorphic/neuroswarm"
    echo "~/neuromorphic/neuroswarm/scripts/hopper/deploy.sh"
    echo
fi
echo "see also: https://github.com/GMU-ASRC/neuroswarm/tree/main/scripts/hopper"

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

rm -rf ~/neuromorphic
{ set +x; } 2>/dev/null  # stop printing commands  https://stackoverflow.com/a/19226038
echo "~/neuromorphic has been scheduled to be removed after this script exits"
echo
echo "So long, and thanks for all the fish"
echo "You may want to log out to unload the previously installed modules"
# https://unix.stackexchange.com/a/33201
(sleep 1; rm -rf ~/neuromorphic) & exit 0