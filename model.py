#Use Keylogger from https://gist.github.com/whym/402801

import cv2, sys, os
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.models import load_model, model_from_json, Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001
MOMENTUM = 0.01
REGULARIZE_PARAM=0.1
DECAY = 0.0

database_loc = "./simulator-linux/driving_log.csv"

DATA_WIDTH = 200
DATA_HEIGHT = 66

train_size = 1.0
show_data = True

# percentage of data to be cut from left, top, right, bottom
ROI_bbox = [0.0, 0.40, 0.0, 0.13]

def cut_ROI_bbox(image_data):
    w = image_data.shape[1]
    h = image_data.shape[0]
    x1 = int(w * ROI_bbox[0])
    x2 = int(w * (1 - ROI_bbox[2]))
    y1 = int(h * ROI_bbox[1])
    y2 = int(h * (1 - ROI_bbox[3]))
    ROI_data = image_data[y1:y2, x1:x2]
    return ROI_data

def preprocess_image(image):
    ROI_data = cut_ROI_bbox(image)
    processed_data = cv2.resize(ROI_data, (DATA_WIDTH, DATA_HEIGHT))
    return processed_data

class ModelHelper:
    def __init__(self, filename):
        self.model = self.define_model()
        self.curr_id = 0
        self.y = list()
        self.x = np.empty((BATCH_SIZE, DATA_HEIGHT, DATA_WIDTH, 3), dtype=np.uint8)
        if(os.path.isfile(filename + ".h5")):
            print "------------------loading model weights-------------"
            self.model = self.load_model_with_weights(filename)

    def define_model(self):
        model = Sequential()
        model.add(Convolution2D(24, 5, 5, name='Conv1', border_mode='same', subsample=(2, 2),
                                input_shape=(DATA_HEIGHT, DATA_WIDTH, 3), activation='relu'))
        model.add(Convolution2D(36, 5, 5, name='Conv2', border_mode='same', subsample=(2, 2)))
        model.add(Convolution2D(48, 5, 5, name='Conv3', border_mode='same', subsample=(2, 2)))
        model.add(Convolution2D(64, 3, 3, name='Conv4', border_mode='same', subsample=(1, 1)))
        model.add(Convolution2D(64, 3, 3, name='Conv5', border_mode='same', subsample=(1, 1)))
        model.add(Flatten(name='Flatten'))
        model.add(Dense(1164, name='Dense1', init='uniform'))
        #model.add(Dropout(0.1))
        model.add(Dense(100, name='Dense2', init='uniform', activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(50, name='Dense3', init='uniform', activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(10, name='Dense4', init='uniform', activation='relu'))
        model.add(Dense(1, name='Dense5', init='uniform', activation='tanh'))
        return model

    def create_random_data(batches):
        x = np.random.random((batch_size * batches, DATA_HEIGHT, DATA_WIDTH, 3))
        y = np.random.randint(2, size=(batch_size * batches, 1))
        return x, y
    
    def read_csv_entries(self, filename):
        self.lines = []
        with open(filename, 'r') as dbfile:
            self.lines = dbfile.readlines()
            #lines.append(dbfile.readlines())
        return self.lines
    
    def get_next_entries(self, count, entries):
        # Allocate twice to accomodate original and flipped images
        x = np.empty((2*count, DATA_HEIGHT, DATA_WIDTH, 3), dtype=np.uint8)
        y = np.empty((2*count), dtype=float)
        idx = 0
        curr = self.curr_id
        for entry in self.lines[curr:curr+count]:
            output = entry.split()
            entry = entry.replace(",", "")
            [c_f, l_f, r_f, steering, throttle, breaks, speed] = entry.split()
            #print steering, c_f
            image_data = cv2.imread(c_f)
            image_data = preprocess_image(image_data)

            #cv2.imshow("Raw Input", image_data)
            #cv2.waitKey(1)
            curr += 1
            if(curr > len(self.lines)):
                curr = 0

            self.curr_id = curr 

            x[idx, :, :, :] = np.copy(image_data)
            y[idx] = np.copy(float(steering))
            idx += 1
        return x, y

    def set_optimizer_params(self, learning_rate):
        Adam_optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)
        self.model.compile(loss='mean_squared_error', optimizer=Adam_optimizer, metrics=['accuracy'])
    
    def start_training(self, x, y):
        #sgd = SGD(lr=LEARNING_RATE) #, momentum=MOMENTUM, decay=DECAY, nesterov=False)
        #self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        hist = self.model.fit(x, y, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
                              validation_split=0.2)
        return hist

    def train_model(self, X_train, y_train):
        h = self.model.fit(X_train, y_train,
            nb_epoch = 1, verbose=0, batch_size=training_batch_size)
        self.model.save_weights(checkpoint_filename)
        print('loss : ',h.history['loss'][-1])
        return model

    def train_with_input(self, x_in, y_in):
        curr_id = len(self.y)
        self.x[curr_id, :, :, :] = x_in
        self.y.append(y_in)
        if len(self.y) == BATCH_SIZE:
            y = np.array(self.y)
            #x = np.array(self.x)
            print("lr: ", self.model.optimizer.get_config()['lr'])
            hist = self.model.fit(self.x, y, nb_epoch=1, batch_size=BATCH_SIZE, verbose=1, validation_data=(self.x, self.y))
            self.y = []
            return hist
        else:
            return None

    def save_model_to_json_file(self, filename):
        json_string = self.model.to_json()
        with open(filename + '.json', 'w') as jfile:
            jfile.write(json_string)
    
    def save_model_weights(self, filename):
        self.model.save(filename + '.h5')
    
    def load_model_from_json(filename):
        with open(filename + '.json', 'r') as jfile:
            json_string = jfile.read()
    
        self.model = model_from_json(json_string)
        return model
    
    def load_model_with_weights(self, filename):
        return load_model(filename + '.h5')
    
    def plot_model_to_file(self, filename):
        plot(self.model, to_file=filename + '.jpg')
    
    def show_model_from_image(self, filename):
        model_image = cv2.imread(filename + ".jpg")
        cv2.imshow("model", model_image)
        cv2.waitKey(0)
    
    def plot_metrics(self, history):
        #import pdb; pdb.set_trace()
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def plot_predictions(self, x, y, x_val, y_val):
        orig_steer = []
        pred_steering_angle = []
        orig_steer = y.tolist()
        for val in y_val:
            orig_steer.append(val)

        for image in x:
            pred_steering_angle.append(float(self.model.predict(image[None, :, :, :], batch_size=1)))
        for image in x_val:
            pred_steering_angle.append(float(self.model.predict(image[None, :, :, :], batch_size=1)))

        print len(orig_steer), len(pred_steering_angle)
        plt.plot(orig_steer)
        plt.plot(pred_steering_angle)
        plt.xlabel('frame')
        plt.ylim([-15, 15])
        plt.ylabel('steering angle')
        plt.legend(['original', 'predicted'], loc='upper right')
        plt.show()

if __name__ == '__main__':
    model_filename = "model.json"
    model_filename = model_filename[:-5]
    model_handle = ModelHelper(model_filename)
    model_handle.plot_model_to_file(model_filename)
    #model_handle.show_model_from_image(model_filename)

    entries = model_handle.read_csv_entries(database_loc)

    train_entries_count = int(len(entries) * train_size)
    x, y = model_handle.get_next_entries(train_entries_count, entries)

    for i in range(train_entries_count):
        x[train_entries_count+i, :, :, :] = cv2.flip(x[i, :, :, :], 1)
        y[train_entries_count+i] = -1.0 * y[i]

    if show_data:
        for i in range(2*train_entries_count):
            cv2.imshow("ROI Input", x[i])
            cv2.waitKey(1)

    print ("Database size: {}".format(len(entries)))

    model_handle.set_optimizer_params(LEARNING_RATE)
    hist = model_handle.start_training(x, y)
    model_handle.save_model_to_json_file(model_filename)
    model_handle.save_model_weights(model_filename)
    #model_handle.plot_metrics(hist)
    model_handle.plot_predictions(x, y, x_val, y_val)
    #sys.exit(0)
