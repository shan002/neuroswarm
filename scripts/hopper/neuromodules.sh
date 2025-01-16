
module load automake
module load autotools
module load readline
module load bzip2
module load sqlite
module load use.own
module load autoconf-2.72
module load libffi-3.4.6
# pyenv doctor

pyenv global 3.12.4

python --version
pip --version

source ~/neuromorphic/framework/pyframework/bin/activate
cd ~/neuromorphic

