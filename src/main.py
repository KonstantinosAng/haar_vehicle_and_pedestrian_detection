"""
Author: Konstantinos Angelopoulos
Date: 08/08/2020

Feel free to use and modify and if you like it give it a star.

AI detect cars and pedestrians using HAAR CASCADES features
inspired by https://www.youtube.com/watch?v=zg9X6ASj3Q0
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Classifier:

  def __init__(self, path):
    # Load video files
    self.video = cv2.VideoCapture(path)
    self.frame = None
    self.gray = None
    self.blur = None
    self.canny = None
    # Load cascade files
    self.car_cascade = 'haar_cascades/car.xml'
    self.ped_cascade = 'haar_cascades/body.xml'
    # create classifiers
    self.car_classifier = cv2.CascadeClassifier(self.car_cascade)
    self.ped_classifier = cv2.CascadeClassifier(self.ped_cascade)

  def detect_lanes(self):
    self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
    low_threshold, upper_threshold = 50, 150
    self.canny = cv2.Canny(self.blur, low_threshold, upper_threshold)
    roi = self.region_of_interest()
    lanes = cv2.HoughLinesP(roi, 2, 1*np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    self.draw_lanes(lanes, (255, 0, 0))

  def region_of_interest(self):
    height, width = self.canny.shape[0], self.canny.shape[1]
    polygons = np.array([[(100, height), (400 + width//2, height), (width//2, 250)]])
    mask = np.zeros_like(self.canny)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(self.canny, mask)
    return masked_image

  def detect_cars(self):
    # detect each car
    cars = self.car_classifier.detectMultiScale(self.gray)
    # place a rectangle around car
    for (x, y, w, h) in cars:
      self.draw_text("Car", (x+w//2, y-5), (0, 0, 255))
      self.draw_rectangle((x, y), (x+w, y+h), (0, 0, 255))

  def detect_pedestrians(self):
    # detect each pedestrian
    peds = self.ped_classifier.detectMultiScale(self.gray)
    # place a rectangle around pedestrians too
    for (x, y, w, h) in peds:
      self.draw_text("Person", (x+w//2, y-5), (0, 255, 0))
      self.draw_rectangle((x, y), (x+w, y+h), (0, 255, 0))

  def draw_rectangle(self, pos1, pos2, color):
    cv2.rectangle(self.frame, pos1, pos2, color, 2)

  def draw_text(self, text, pos, color):
    cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

  def draw_lanes(self, lanes, color):
    lane_image = np.zeros_like(self.frame)
    if lanes is not None:
      for line in lanes:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(lane_image, (x1, y1), (x2, y2), color, 10)
    self.frame = cv2.addWeighted(lane_image, 0.8, self.frame, 1, 1)

  def detect(self):
    skip = 0
    # open video
    while self.video.isOpened():
      # start reading video frames
      ret, self.frame = self.video.read()
      # if next frame grabbed
      if ret and skip%3 == 0:
        # transform to gray
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.detect_cars()
        self.detect_pedestrians()
        self.detect_lanes()
        # show frame
        cv2.imshow('DETECTION', cv2.resize(self.frame, (960, 540)))
        # listen for keys
        key_pressed = cv2.waitKey(1)

      if key_pressed == 81 or key_pressed == 113:
        break 
      
      skip += 1

    self.video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  from optparse import OptionParser
  parser = OptionParser()
  parser.add_option('--video', action='store', default='videos/ped.mp4', type='string',
                    dest='video', help='path to the video file')

  (options, args) = parser.parse_args()

  if options.video:
    video_file = options.video
    classifier = Classifier(video_file)
    classifier.detect()
