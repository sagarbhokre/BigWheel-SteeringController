#!/bin/python

STEERING_FILEPATH='./steering.csv'
IMAGE_DIRPATH='./center'
DESTINATION_FILEPATH='./data.txt'
data_scaling_factor = 1000

import csv
import math

from os import listdir
from os.path import isfile, join

def main():
    # (1) Read CSV steering file
    #     Find Min / Max

    steeringList = list()
    angle = list()
    
    with open(STEERING_FILEPATH, 'rb') as csvfile:
        steering_data = csv.reader(csvfile, delimiter=',')
        
        for i,row in enumerate(steering_data):
            if i == 0:
                # Skip Header
                # seq,timestamp,angle,torque,speed
                continue
            pack = (int(row[1]), float(row[2]), float(row[4]))
            angle.append(float(row[2])) # * 180 / math.pi)
            
            steeringList.append(pack)

    steeringList.sort()
    
    #print "Angle: ", min(angle), max(angle)

    # (2) Read all filenames from IMAGE_DIRPATH
    #     Match timestamp with filenames (timestamp); 
    #     Fetch the most recent filename

    onlyfiles = [int(f[:-4]) for f in listdir(IMAGE_DIRPATH) if isfile(join(IMAGE_DIRPATH, f))]
    onlyfiles.sort()

    stidx = 1

    data = list()

    for timestamp in onlyfiles:

        if stidx == len(steeringList):
            break
       
        # Matching Timestamps across two sorted lists
        while (steeringList[stidx][0] <= timestamp):
            stidx += 1

        # Exess steering movement may not be suitable for steering control training set
        if ((steeringList[stidx][1] > 1) or (steeringList[stidx][1] < -1)):
            continue

        # Speed < 1
        if (steeringList[stidx][2] < 1):
            continue

        latest_steering_angle = steeringList[stidx-1][1]

        # (3) Data Structure:
        #     image_filename(timestamp.jpg), matched steering angle in degrees, 

        img_file = str(timestamp) + '.jpg'

        pack = (img_file, latest_steering_angle * data_scaling_factor)

        data.append(pack)

    #print data[:3], data[-3:]    

    # (4) Save Data Structure into a file
    for filename, angle in data:
        print filename, angle
        


main()

