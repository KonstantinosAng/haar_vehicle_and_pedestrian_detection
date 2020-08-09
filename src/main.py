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
    try:
      self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
      low_threshold, upper_threshold = 50, 150
      self.canny = cv2.Canny(self.blur, low_threshold, upper_threshold)
      roi = self.region_of_interest()
      lanes = cv2.HoughLinesP(roi, 2, 1*np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
      lanes = self.average_slope(lanes)
      self.draw_lanes(lanes, (255, 0, 0))
    except Exception as e:
      print('[CANT DETECT LANES] {}'.format(e))

  def region_of_interest(self):
    height, width = self.canny.shape[0], self.canny.shape[1]
    polygons = np.array([[(int(width//2 - width//2.1), height-100), (int(width//2 + width//2.1), height-100), (width//2, height//2 + 110)]])
    mask = np.zeros_like(self.canny)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(self.canny, mask)
    # cv2.imshow('mask', cv2.resize(masked_image, (960, 540)))
    return masked_image

  def average_slope(self, lanes):
    try:
      left, right = [], []
      for line in lanes:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0], parameters[1]
        if slope < 0:
          left.append((slope, intercept))
        else:
          right.append((slope, intercept))
      left_average, right_average = np.average(left, axis=0), np.average(right, axis=0)
      if left_average.size == 1: left_average = np.array([1, 1])
      if right_average.size == 1: right_average = np.array([1, 1])
      right_line = self.fix_lane_coordinate(right_average)
      left_line = self.fix_lane_coordinate(left_average)
      return np.array([left_line, right_line])
    except:
      return np.array([1, 1, 1, 1])

  def fix_lane_coordinate(self, average):
    try:
      slope, intercept = average
      y1 = self.frame.shape[0]
      y2 = int(y1*(3/5))
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
      return np.array([x1, y1, x2, y2])
    except:
      return np.array([1, 1, 1, 1])

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
    try:
      lane_image = np.zeros_like(self.frame)
      if len(lanes) > 1:
        for line in lanes:
          x1, y1, x2, y2 = line
          #cv2.line(lane_image, (x1, y1), (x2, y2), color, 10)
          cv2.polylines(lane_image, [np.array([[x1, y1], [x2, y2]])], True, color, 10)
        self.frame = cv2.addWeighted(lane_image, 0.8, self.frame, 1, 1)
        # shade lanes
        x1, y1, x2, y2 = lanes[0]
        x3, y3, x4, y4 = lanes[1]
        polygons = np.array([[[x1, y1], [x2, y2], [x4, y4], [x3, y3]]])
        cv2.fillPoly(self.frame, polygons, color=[160, 32, 240])
    except Exception as e:
      print('[CANT DRAW LANES] {}'.format(e))

  def detect(self):      
    skip = 0
    # open video
    while self.video.isOpened():
      try:
        # start reading video frames
        ret, self.frame = self.video.read()
        # if next frame grabbed
        if ret and skip%6 == 0:
          # transform to gray
          self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
          self.detect_cars()
          self.detect_pedestrians()
          self.detect_lanes()
          # show frame
          cv2.imshow('DETECTION', cv2.resize(self.frame, (960, 540)))
          # listen for keys
          key_pressed = cv2.waitKey(1)
          skip = 0
        
          if key_pressed == 81 or key_pressed == 113:
            break
        
        if not ret:
          break
        
        skip += 1

      except:
        pass

    self.video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  from optparse import OptionParser
  parser = OptionParser()
  parser.add_option('-v', action='store', default='videos/test.mp4', type='string',
                    dest='video', help='path to the video file')

  (options, args) = parser.parse_args()

  if options.video:
    video_file = options.video
    classifier = Classifier(video_file)
    classifier.detect()
