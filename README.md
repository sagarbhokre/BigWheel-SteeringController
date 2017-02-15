BigWheel-SteeringController
========

BigWheel is an Artificially Intelligent (AI) steering wheel controller which mimics human behavior. A baby on wheels to start with which evolves as it sees how we drive and learns as we drive. BigWheel confines its Region Of Interest and uses a neural network to learn how to maneuver on roads. All that is needed is a network initialized with random weights and around 1 hour of training to refine the performance to a level that BigWheel could potentially offload your daily mundane commute. On-The-Go training feature is a unique feature of this implementation and this AI module has a potential to improve with experience and drive the way you drive.


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

This network is known to work well for steering control and hence this was selected as a primary model for evaluation and development

**Model Architecture**

![Network Architecture Image](https://github.com/sagarbhokre/BigWheel-SteeringController/blob/master/model.jpg)

Neural network using in the implementation uses following layers:
- Conv1 layer (24 kernels of size 5x5 and stride 2 in both x and y direction)
- Conv2 layer (36 kernels of size 5x5 and stride 2 in both x and y direction)
- Conv3 layer (48 kernels of size 5x5 and stride 2 in both x and y direction)
- Conv4 layer (64 kernels of size 3x3 and stride 1 in both x and y direction)
- Conv5 layer (64 kernels of size 3x3 and stride 1 in both x and y direction)
- Dense1 layer (1164 neurons fully connected and passed through relu activation)
- Dense2 layer (100 neurons fully connected and passed through relu activation)
- Dense3 layer (50 neurons fully connected and passed through relu activation)
- Dense4 layer (10 neurons fully connected and passed through relu activation)
- Dense5 layer (1 neuron fully connected computing inverse tan)

While training the network, data is internally split into 80% for training and 20% for validation

Adam optimizer is used to train the network

Since the network is trained On-The-Go and since it is cloning valid driving, there was no need to prune incoming data.

**Initial Training:**

To train the network for initial stage, Images were captured using "Training mode" of the simulator for about 1 min and then the model was trained using model.py script

model.py is a reference model trainer. This script serves only as a starting point for training the network (starting point for model's weights and biases)

Record a set of images and steering angles using training mode of simulator and use this script to train the model. It is fine to train the network with 5-10 sec worth of data. It does not matter if the training happened with invalid data. Once initial training is done the model is refined and tuned on the training track using drive.py script.(Explained ahead)

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
