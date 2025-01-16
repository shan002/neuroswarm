#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"  # cd to the location of this script

set -e  # EXIT ON ERROR

mkdir -p ~/privatemodules
cp -Rf privatemodules/** ~/privatemodules  # copy privatemodules/ to $HOME

# make project directory
mkdir -p ~/neuromorphic
cd ~/neuromorphic

# download the relevant repositories
if [ ! -d ~/neuromorphic/RobotSwarmSimulator ]; then
  git clone https://github.com/kenblu24/RobotSwarmSimulator.git
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
  git clone git@bitbucket.org:neuromorphic-utk/framework.git
fi
if [ ! -d ~/neuromorphic/framework/processors/caspian ]; then
  git clone git@bitbucket.org:neuromorphic-utk/caspian.git ~/neuromorphic/framework/processors/caspian
fi

# make a spot for our python local dependencies to go
mkdir -p ~/.local/packages
cd ~/.local/packages
PACKAGES=`realpath '.'`

# luckily hopper has these dependencies available already; no need to build them
# let's load them now
module load automake
module load autotools
module load readline
module load bzip2
module load sqlite

# sadly autoconf and libffi are dependencies for python.
# libffi is needed for _ctypes which is needed for scipy

# download and build autoconf-2.72
cd ~/.local
wget https://ftp.wayne.edu/gnu/autoconf/autoconf-2.72.tar.xz
tar -xf autoconf-2.72.tar.xz
cd autoconf-2.72
./configure --prefix $PACKAGES/autoconf-2.72
make && make install

module load use.own  # load this to update the available user modules
module load autoconf-2.72

# download and build libffi
cd ~/.local
wget https://github.com/libffi/libffi/releases/download/v3.4.6/libffi-3.4.6.tar.gz
tar -xf libffi-3.4.6.tar.gz
cd libffi-3.4.6
./configure --prefix $PACKAGES/libffi-3.4.6
make && make install

module load use.own  # load this to update the available user modules
module load libffi-3.4.6

cd

# download and install pyenv if we don't have it
if ! command -v pyenv &> /dev/null
then
	rm -rf ~/.pyenv  # if ~/.pyenv exists, the install script will error

	curl https://pyenv.run | bash

	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc

	source ~/.bashrc
fi

pyenv doctor  # check if pyenv thinks we can build python

pyenv install 3.12.4 --force

pyenv global 3.12.4  # load our newly-built python

# make sure we can run python and pip
python --version
pip --version

set -x  # print commands
# check that we have _ctypes for scipy
python -c 'import _ctypes'
{ set +x; } 2>/dev/null  # stop printing commands

# make the pyframework virtual environment manually
# the framework/scripts/create_env.sh script uses `venv`` which is less versatile
# so we run `virtualenv`` ourselves to make it ahead of time
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
cd ~/neuromorphic/RobotSwarmSimulator
pip install -r mindeps.txt
pip install -e .

# install other dependencies
cd ~/neuromorphic/turtwig
pip install -r requirements.txt
cd ~/neuromorphic
unalias pip

set -x  # print commands
# check that these work
python -c 'import neuro'
python -c 'import caspian'
python -c 'import novel_swarms'
{ set +x; } 2>/dev/null  # stop printing commands  https://stackoverflow.com/a/19226038

echo Everything seems to be working!
echo "Next time you login, don't forget to enable the modules:"
echo ">>>  source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh"
