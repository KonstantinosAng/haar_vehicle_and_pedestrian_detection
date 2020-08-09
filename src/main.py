"""
Author: Konstantinos Angelopoulos
Date: 08/08/2020

Feel free to use and modify and if you like it give it a star.

AI detect cars and pedestrians using HAAR CASCADES features
inspired by https://www.youtube.com/watch?v=zg9X6ASj3Q0
"""

import cv2

class Classifier:

  def __init__(self, path):
    # Load video files
    self.video = cv2.VideoCapture(path)
    self.frame = None
    self.gray = None
    # Load cascade files
    self.car_cascade = 'haar_cascades/car.xml'
    self.ped_cascade = 'haar_cascades/body.xml'
    # create classifiers
    self.car_classifier = cv2.CascadeClassifier(self.car_cascade)
    self.ped_classifier = cv2.CascadeClassifier(self.ped_cascade)

  def detect_lanes(self):
    pass

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

  def detect(self):
    # open video
    while self.video.isOpened():
      # start reading video frames
      ret, self.frame = self.video.read()
      # if next frame grabbed
      if ret:
        # transform to gray
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.detect_cars()
        self.detect_pedestrians()
        # show frame
        cv2.imshow('DETECTION', self.frame)
        # listen for keys
        key_pressed = cv2.waitKey(1)

      if key_pressed == 81 or key_pressed == 113:
        break 

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
    classifier.run()
