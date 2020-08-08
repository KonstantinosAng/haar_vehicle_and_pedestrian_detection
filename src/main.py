""" AI detect cars and pedestrians using HAAR CASCADES features """

import cv2

# Load video files
ped_video = cv2.VideoCapture('videos/ped.mp4')
# Load cascade files
car_cascade = 'haar_cascades/car.xml'
ped_cascade = 'haar_cascades/body.xml'

# create classifiers
car_classifier = cv2.CascadeClassifier(car_cascade)
ped_classifier = cv2.CascadeClassifier(ped_cascade)

# open video
while ped_video.isOpened():
  # start reading video frames
  ret, frame = ped_video.read()
  # if next frame grabbed
  if ret:
    # transform to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect each car
    cars = car_classifier.detectMultiScale(gray)
    # detect each pedestrian
    peds = ped_classifier.detectMultiScale(gray)
    # place a rectangle around car
    for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # place a rectangle around pedestrians too
    for (x, y, w, h) in peds:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # show frame
    cv2.imshow('DETECTION', frame)
    # listen for keys
    key_pressed = cv2.waitKey(1)

  if key_pressed == 81 or key_pressed == 113:
    break 

ped_video.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
  from option
  