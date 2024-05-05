sudo apt-get update
sudo apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev libopencv-dev python3-setuptools python3-dev python3-pip libhdf5-dev
# pip3 install opencv-python==4.1.0.25 torch==1.1.0 torchvision==0.3.0 numpy==1.13.3 pandas==0.24.1 networkx==2.2

# cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/NaviLLM
# source activate 
# conda activate navillm

/usr/local/bin/pip3 install --upgrade pip setuptools wheel
/usr/local/bin/pip3 install easydict==1.10 jsonlines==2.0.0 lmdb==1.4.1 more_itertools==10.1.0 msgpack_numpy==0.4.8 msgpack_python==0.5.6 networkx==2.5.1 numpy==1.22.3 opencv_python==4.7.0.72 Pillow==10.1.0 progressbar33==2.4 psutil==5.9.4 PyYAML==6.0.1  ray==2.8.0 requests==2.25.1 shapely==2.0.1 timm==0.9.2 tqdm==4.64.1 transformers==4.28.0 sentencepiece==0.1.99 cython
/usr/local/bin/pip3 install --use-pep517 h5py
/usr/local/bin/pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117 
/usr/local/bin/pip3 install peft==0.3.0
/usr/local/bin/pip3 install wandb

PYTHONPATH=/mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator_bak/build:$PYTHONPATH
cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator_bak
# mkdir build && cd build
cd build
# cmake -DEGL_RENDERING=ON ..
make
cd ../

----------------

cp -r /mnt/bn/kinetics-lp-maliva-v6/data/VLN /home/tiger/

----------------
cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/NaviLLM
# source activate 
# conda activate navillm

----------------

/usr/local/bin/pip3 install --upgrade pip setuptools wheel
/usr/local/bin/pip3 install easydict==1.10 jsonlines==2.0.0 lmdb==1.4.1 more_itertools==10.1.0 msgpack_numpy==0.4.8 msgpack_python==0.5.6 networkx==2.5.1 numpy==1.22.3 opencv_python==4.7.0.72 Pillow==10.1.0 progressbar33==2.4 psutil==5.9.4 PyYAML==6.0.1  ray==2.8.0 requests==2.25.1 shapely==2.0.1 timm==0.9.2 tqdm==4.64.1 transformers==4.28.0 sentencepiece==0.1.99 cython
/usr/local/bin/pip3 install --use-pep517 h5py
/usr/local/bin/pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117 
/usr/local/bin/pip3 install peft==0.3.0
/usr/local/bin/pip3 install wandb

----------------

cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator_bak

#sometimes need to rm build folder, and rebuild twice
#export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH

mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../

PYTHONPATH=/mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator_bak/build:$PYTHONPATH
export JAVA_HOME=/opt/tiger/yarn_deploy/jdk
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar