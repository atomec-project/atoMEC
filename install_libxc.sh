#!/bin/bash

# path to where libxc should be installed
libxc_path=~
cd $libxc_path

# path to where libxc shared libraries should be stored
install_path=$libxc_path/libxc/sharedlib/

# clone the repo
git clone git@github.com:ElectronicStructureLibrary/libxc.git
cd libxc
mkdir -p $install_path

# build with cmake
cmake -H. -Bobjdir -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$install_path -DENABLE_PYTHON=ON
cd objdir && make
make test
make install

# Add libxc shared lib path to install_path
if ! grep -q "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$install_path" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$install_path" >> ~/.bashrc
fi

# install the python bindings
cd $libxc_path/libxc
pip install .

echo "libxc installation complete"





