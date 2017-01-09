SteeringController
========

Neural network to control steering angle

Preview of execution: https://youtu.be/4mPIFE48cmk

[![EXECUTION_PREVIEW](http://img.youtube.com/vi/4mPIFE48cmk/0.jpg)](http://www.youtube.com/watch?v=4mPIFE48cmk "Behavioral cloning using BigWheel trainer")

**Installation instructions:**
----------


**Download simulator:**

wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip

mkdir simulator-linux

unzip simulator-linux.zip -d simulator-linux

mv simulator-linux.zip simulator-linux


**Run simulator:**

./run_simulator &

**Install dependencies:**

sudo apt-get install python-pip libhdf5-dev python-opencv python-xlib graphviz

sudo pip install eventlet

sudo pip install numpy

sudo pip install flask-socketio

sudo pip install eventlet

sudo pip install pillow

sudo pip install h5py

sudo pip install keras

sudo pip install pydot==1.1.0

sudo pip install graphviz

sudo pip install tensorflow-gpu

**Execute the neural network**

To run the network, refer to the script: start_network

To stop the network, refer to the script: kill_network


Implementation details:
=======================

Neural Network details:
-----------------------

Reference network used to implement the steering wheel controller is [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf)

**Model Architecture**

![Network Architecture](https://github.com/sagarbhokre/BigWheel-SteeringController/blob/master/model.jpg)

**Initial Training:**
model.py is a reference model trainer. This script serves only as a starting point for training the network. 
This script is used to set a starting point for model's weights and biases. Record a set of images and steering angles using training mode of simulator and use this script to train the model. It is fine to train the network with 5-10 sec worth of data. It does not matter if the training happened with invalid data. Once initial training is done the model is refined and tuned on the training track. Actual training happens with the help of script drive.py (Explained ahead)

**Currently generator is not being used to retrieve batches of data as this is an initial step**

**On-The-Go training:**

**Keyboard mapping**:

<_Up Arrow_>: Increase throttle

<_Down Arrow_>: Decrease throttle

<_Left Arrow_>: Turn left

<_Right Arrow_>: Turn right

'X' or 'x': Toggle manual override state

'R' or 'r': Reload last saved model from file

'S' or 's': Save current state of the model to file

<_Esc_>: Exit application


**Start simulator** (_refer to run_simulator script for instructions_)

Launch neural network based inference/training engine (_refer to start_network script for instructions_)

Once the simulator is up and network is running, go to "Autonomous mode"

**Evaluation of network**:

Start simulator and run neural network using drive.py. The network should accept image input from the simulator and generate steering angles accordingly. Please do not press any keys mentioned in "keyboard mapping" this would mean inference without human intervention.

With the model available in the repository, performance is as seen in the youtube video.

**Refining performance of the network**:

To refine the performance, enter manual override mode by pressing 'x' maneuver the vehicle to correct position and then press 'x' again to exit manual override mode.

Repeat above step until the behavior is acceptable.
