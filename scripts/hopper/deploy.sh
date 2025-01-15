#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

set -e  # exit on error

mkdir -p ~/privatemodules
cp -Rf privatemodules/** ~/privatemodules

mkdir -p ~/neuromorphic
cd ~/neuromorphic

if [ ! -d ~/neuromorphic/RobotSwarmSimulator ]; then
  git clone https://github.com/kenblu24/RobotSwarmSimulator.git
fi

if [ ! -d ~/neuromorphic/framework ]; then
  git clone git@bitbucket.org:neuromorphic-utk/framework.git
fi

if [ ! -d ~/neuromorphic/framework/processors/caspian ]; then
  git clone git@bitbucket.org:neuromorphic-utk/caspian.git ~/neuromorphic/framework/processors/caspian
fi


mkdir -p ~/.local/packages
cd ~/.local/packages
PACKAGES=`realpath 'packages'`

module load automake
module load autotools
module load readline
module load bzip2
module load sqlite

cd ~/.local
wget https://ftp.wayne.edu/gnu/autoconf/autoconf-2.72.tar.xz
tar -xf autoconf-2.72.tar.xz
cd autoconf-2.72
./configure --prefix $PACKAGES/autoconf-2.72
make && make install

module load use.own
module load autoconf-2.72

cd ~/.local
wget https://github.com/libffi/libffi/releases/download/v3.4.6/libffi-3.4.6.tar.gz
tar -xf libffi-3.4.6.tar.gz
cd libffi-3.4.6
./configure --prefix $PACKAGES/libffi-3.4.6
make && make install

cd

module load use.own
module load libffi-3.4.6

if ! command -v pyenv &> /dev/null
then
	curl https://pyenv.run | bash

	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc

	source ~/.bashrc
fi

pyenv doctor

pyenv install 3.12.4 --force

pyenv global 3.12.4

python --version
pip --version

mkdir -p ~/neuromorphic/framework
cd ~/neuromorphic/framework
pip install virtualenv
python -m virtualenv pyframework
source pyframework/bin/activate
pip install uv
shopt -s expand_aliases
alias pip='uv pip'
source ./scripts/create_env.sh
cd ~/neuromorphic/RobotSwarmSimulator
pip install -r mindeps.txt
pip install -e .

cd ~/neuromorphic/turtwig
pip install -r requirements.txt
cd ~/neuromorphic
unalias pip

set -x
python -c 'import neuro'
python -c 'import caspian'
python -c 'import novel_swarms'
set +x

echo Everything seems to be working!
echo "Next time you login, don't forget to enable the modules:"
echo ">>>  source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh"
