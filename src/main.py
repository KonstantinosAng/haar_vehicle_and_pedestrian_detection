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
    self.ped_video = cv2.VideoCapture(path)
    # Load cascade files
    self.car_cascade = 'haar_cascades/car.xml'
    self.ped_cascade = 'haar_cascades/body.xml'
    # create classifiers
    self.car_classifier = cv2.CascadeClassifier(self.car_cascade)
    self.ped_classifier = cv2.CascadeClassifier(self.ped_cascade)

  def detect(self):
    # open video
    while self.ped_video.isOpened():
      # start reading video frames
      ret, frame = self.ped_video.read()
      # if next frame grabbed
      if ret:
        # transform to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect each car
        cars = self.car_classifier.detectMultiScale(gray)
        # detect each pedestrian
        peds = self.ped_classifier.detectMultiScale(gray)
        # place a rectangle around car
        for (x, y, w, h) in cars:
          cv2.putText(frame,"Car", (x+w//2, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # place a rectangle around pedestrians too
        for (x, y, w, h) in peds:
          cv2.putText(frame,"Person", (x+w//2, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # show frame
        cv2.imshow('DETECTION', frame)
        # listen for keys
        key_pressed = cv2.waitKey(1)

      if key_pressed == 81 or key_pressed == 113:
        break 

    self.ped_video.release()
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
