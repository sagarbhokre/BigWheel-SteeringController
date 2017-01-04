#!/usr/bin/env python
import argparse
import base64
import json
import ipc
import signal, sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from model_helper import *
from keyboard_monitor import *

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

ENABLE_SOCKET = False

sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None

G_MAX_STEERING_ANGLE = 0.3
g_steering_angle = 0.0
g_throttle = 0.1
g_enable_override = False
g_kill = False

class DriveController:
    def __init__(self, modeljsonfile):
        self.hModel = ModelHelper(modeljsonfile[:-5])
        self.model = self.hModel.model;
        self.hModel.set_optimizer_params(learning_rate=0.0001) #compile("adam", "mse") #sgd

        self.kb_cb = keyboard_callback
        self.kb_ctx = initialize_keyboard_monitor()
        self.start_kb_thread()

    def start_kb_thread(self):
        self.kb_thread = threading.Thread(name='Keyboard_Thread', target=self.kb_event_loop)
        self.kb_thread.start()

    def stop_kb_thread(self):
        stop_keyboard_monitor(self.kb_ctx)
        self.kb_thread.join(0.2)

    def kb_event_loop(self):
        start_keyboard_monitor(self.kb_ctx, self.kb_cb)
        # control should come here after kb context is disabled after ctrl+c
        free_keyboard_ctx(self.kb_ctx)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    image = preprocess_image(image_array)
    cv2.imshow("Input to n/w", image)
    cv2.waitKey(1)

    transformed_image = image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    global g_steering_angle
    if g_enable_override :
        steering_angle = g_steering_angle
        print ("Steering angle: ", steering_angle)
        hDrive.hModel.train_with_input(image, steering_angle)
    else:
        steering_angle = float(hDrive.model.predict(transformed_image, batch_size=1))
        g_steering_angle = float(steering_angle)

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = g_throttle
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

'''
def train_with_input(image, steering_angle):
    y = list()
    x = list()
    y.append(steering_angle)
    x.append(image[None, :, :, :])
    print("Train with : ", y, x[0].shape)
    hist = hDrive.model.fit(x, np.array(y), nb_epoch=1, batch_size=1) #, validation_data=(image, steering_angle))
    print hist
    return hist
'''

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


class Event(ipc.Message):
    def __init__(self, event_type, **properties):
        self.type = event_type
        self.properties = properties

    def _get_args(self):
        return [self.type], self.properties


class Response(ipc.Message):
    def __init__(self, text):
        self.text = text

    def _get_args(self):
        return [self.text], {}

def get_mouse_loc(obj):
    x = obj.properties['x']
    y = obj.properties['y']
    return x, y

def get_key_value(obj):
    val = obj.properties['key']
    return val

def server_process_request(objects):
    try:
        for obj in objects:
            obj = objects[0]
            if(obj.type == 'mouse'):
                x, y = get_mouse_loc(obj)
                print "Mouse event: (" + x + ", " + y + ")"
            elif(obj.type == 'keyboard'):
                val = get_key_value(obj)
                print "Keyboard event: (" + val + ")"
    except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    response = [Response('Recieved {} objects'.format(len(objects)))]
    print 'Recieved objects: {}'.format(objects)
    print 'Sent objects: {}'.format(response)
    return response

import threading
def ipc_event_loop(server_address, kwargs):
    server_address = (server_address, kwargs)
    print ("IPC event loop : " + str(server_address) + "--" + str(kwargs))
    #Start IPC server to receive keyboard/mouse events
    ipc.Server(server_address, server_process_request).serve_forever()

def keyboard_callback(key):
    #print "keyboard callback :" + key
    global g_steering_angle, g_enable_override, g_throttle
    if key == 'a' or key == 'Right':
        g_steering_angle += 0.1
        if g_steering_angle > G_MAX_STEERING_ANGLE:
            g_steering_angle = G_MAX_STEERING_ANGLE
    if key == 'd' or key == 'Left' :
        g_steering_angle -= 0.1
        if g_steering_angle < -G_MAX_STEERING_ANGLE:
            g_steering_angle = -G_MAX_STEERING_ANGLE
    if key == 'w' or key == 'Up':
        g_throttle += 0.1
        if g_throttle > 1.0:
            g_throttle = 1.0
    if key == 's' or key == 'Down' :
        g_throttle -= 0.1
        if g_throttle < -1.0:
            g_throttle = -1.0
    if key == 'x' or key == 'X' :
        g_enable_override = not g_enable_override
    if key == 'r' or key == 'R' :
        g_reset_model = 1
        print("Model reload")
        hDrive.model = hDrive.hModel.load_model_with_weights(args.model[:-5])
    if key == 's' or key == 'S' :
        print("Model saved")
        hDrive.hModel.save_model_to_json_file(args.model[:-5])
        hDrive.hModel.save_model_weights(args.model[:-5])
    if key == 'Escape' :
        g_kill = True
    #print "Steering: " + str(g_steering_angle * 25) + " Throttle: " + str(g_throttle)

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    hDrive.stop_kb_thread()
    print('Waiting for sys exit')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--socket', type=str,
        help='Location of socket file [default: None] from where keyboard/mouse events can be receved.')
    parser.add_argument('--host', type=str,
        help='Location of host [default: localhost] from where keyboard/mouse events can be receved.')
    parser.add_argument('--port', type=str,
        help='Port address [default: 5795] at which IPC communications should happen.')

    args = parser.parse_args()

    hDrive = DriveController(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    if ENABLE_SOCKET :
        socket = None
        if (args.socket != ""):
            socket = args.socket

        host = 'localhost'
        if(args.host != None):
            host = args.host

        port = 5795
        if(args.port != None):
            port = int(args.port)

        print "=="+str(args.port)+ "=="
        print host+":"+str(port)

        server_address = socket or (host, port)

        print server_address
        event_thread = threading.Thread(name='event_thread', target=ipc_event_loop, args=server_address)
        event_thread.start()

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    if ENABLE_SOCKET :
        event_thread.join(0.3)

    sys.exit(0)
