#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
PYTHON_VERSION=${PYTHON_VERSION:-3.13}

cd "$parent_path"  # cd to the location of this script

set -e  # EXIT ON ERROR

# make project directory
mkdir -p ~/neuromorphic
cd ~/neuromorphic

# download the relevant repositories
if [ ! -d ~/neuromorphic/RobotSwarmSimulator ]; then
	echo "Downloading RobotSwarmSimulator from github"
  git clone https://github.com/kenblu24/RobotSwarmSimulator.git
else
	echo "RobotSwarmSimulator dir exists, skipping download"
fi

# This next bit downloads the TENNLAB framework.
# You'll need to have your SSH key in bitbucket for this to work.
# You may need to generate a new `id_rsa` or `id_ed25519` key pair
# and modify ~/.ssh/config for git to use the correct key.
#
# Here's an example ~/.ssh/config file:
#
# Host *
#    IdentityFile ~/.ssh/id_rsa
#    StrictHostKeyChecking=no
#    UserKnownHostsFile ~/.ssh/known_hosts
#
if [ ! -d ~/neuromorphic/framework ]; then
	echo "Attempting to download TennLab framework from bitbucket"
  git clone git@bitbucket.org:neuromorphic-utk/framework.git
else
	echo "TennLab framework already downloaded"
fi
if [ ! -d ~/neuromorphic/framework/processors/caspian ]; then
	echo "Downloading ORNL caspian from bitbucket"
  git clone git@bitbucket.org:neuromorphic-utk/caspian.git ~/neuromorphic/framework/processors/caspian
fi

# make a spot for our python local dependencies to go
mkdir -p ~/.local/packages
cd ~/.local/packages
PACKAGES=`realpath '.'`

cd

if [ ! -d ~/.pyenv ]; then
	echo
	echo "You don't have pyenv installed."
	echo "You can install it with:"
	echo
	echo "git clone https://gitlab.orc.gmu.edu/kzhu4/hoppertools.git ~/.hoppertools"
	echo "cd ~/.hoppertools && make pyenv"
	echo
elif command -v pyenv &> /dev/null; then
	source ~/privatemodules/load_pyenv.sh
fi


pyenv doctor  # check if pyenv thinks we can build python

PYTHON_VERSION_ACTUAL="$(pyenv latest -k $PYTHON_VERSION)"

echo
echo "Python version $PYTHON_VERSION_ACTUAL will be installed."
echo "Installing Python now. This might take a while..."
pyenv install $PYTHON_VERSION_ACTUAL --force

pyenv global $PYTHON_VERSION_ACTUAL  # load our newly-built python

# make sure we can run python and pip
python --version
pip --version
echo "Python installed successfully!"
echo
echo "Checking if we can import _ctypes (requires libffi) (scipy needs this)"
set -x  # print commands
# check that we have _ctypes for scipy
python -c 'import _ctypes'
{ set +x; } 2>/dev/null  # stop printing commands

# make the pyframework virtual environment manually
# the framework/scripts/create_env.sh script uses `venv`` which is less versatile
# so we run `virtualenv`` ourselves to make it ahead of time
echo
echo "Building & Installing framework"
echo
mkdir -p ~/neuromorphic/framework
cd ~/neuromorphic/framework
pip install virtualenv
python -m virtualenv pyframework
source pyframework/bin/activate
pip install uv  # uv speeds package install up a LOT but sometimes errors when regular pip still works
shopt -s expand_aliases  # make aliases work in our shell
alias pip='uv pip'  # use uv instead of pip for the following python package installs
source ./scripts/create_env.sh  # install framework and its dependencies
if [ -z ${VIRTUAL_ENV+x} ];  # if no virtual environment detected (if create_env deactivated the venv)
then
	source pyframework/bin/activate
fi

# install RSS and its dependencies
echo
echo "Installing RobotSwarmSimulator and its dependencies"
echo
cd ~/neuromorphic/RobotSwarmSimulator
pip install -r mindeps.txt
pip install -e .

# install other dependencies
echo
echo "Installing dependencies for neuromorphic_experiments (turtwig)"
echo
cd "$parent_path"
cd ../..
pip install -r requirements.txt
cd ~/neuromorphic
unalias pip

echo
echo "Testing if we can import what we just installed"
echo
set -x  # print commands
python -c 'import neuro'
python -c 'import caspian'
python -c 'import swarmsim'
{ set +x; } 2>/dev/null  # stop printing commands  https://stackoverflow.com/a/19226038
echo
echo Everything seems to be working!
echo "Next time you login, don't forget to enable the modules:"
echo ">>>  source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh"
