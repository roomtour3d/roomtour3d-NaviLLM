sudo apt-get update
sudo apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip
pip3 install opencv-python==4.1.0.25 torch==1.1.0 torchvision==0.3.0 numpy==1.13.3 pandas==0.24.1 networkx==2.2

cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/NaviLLM
source activate 
conda activate navillm3

PYTHONPATH=/mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator/build:$PYTHONPATH
cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../

----------------

cp -r /mnt/bn/kinetics-lp-maliva-v6/data/VLN /home/tiger/

----------------
cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/NaviLLM
source activate 
conda activate navillm3

cd /mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator

#sometimes need to rm build folder, and rebuild twice

mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../

PYTHONPATH=/mnt/bn/kinetics-lp-maliva-v6/playground_projects/Matterport3DSimulator/build:$PYTHONPATH
export JAVA_HOME=/opt/tiger/yarn_deploy/jdk
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar