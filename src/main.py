"""
Author: Konstantinos Angelopoulos
Date: 08/08/2020

Feel free to use and modify and if you like it give it a star.

AI detect cars, pedestrians and lanes using HAAR CASCADES features and image process
inspired by https://www.youtube.com/watch?v=zg9X6ASj3Q0
and https://github.com/tatsuyah/Lane-Lines-Detection-Python-OpenCV
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
    self.cars = None
    self.peds = None
    self.lane_img = None
    self.destination_image = None
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
      return lanes
    except Exception as e:
      print('[CANT DETECT LANES] {}'.format(e))

  def region_of_interest(self):
    height, width = self.canny.shape[0], self.canny.shape[1]
    polygons = np.array([[(int(width//2 - width//2.1), height-60), (int(width//2 + width//2), height-60), (width//2, height//2 + 100)]])
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
      return np.array([[1, 1, 1, 1]])

  def fix_lane_coordinate(self, average):
    try:
      slope, intercept = average
      y1 = self.frame.shape[0]
      y2 = int(y1*(3/5))
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
      return np.array([x1, y1, x2, y2])
    except:
      return np.array([[1, 1, 1, 1]])

  def hsl_detection(self):
    sx_thresh=(15, 255)
    s_thresh=(100, 255)
    img = np.copy(self.lane_img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

  def perspective_warp_top_view(self):
    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    img_size = np.float32([(self.destination_image.shape[1], self.destination_image.shape[0])])
    src *= img_size
    dst *= np.float32((self.frame.shape[1], self.frame.shape[0]))
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(self.destination_image, M, (self.frame.shape[1], self.frame.shape[0]))
  
  def inv_perspective_warp_top_view(self, img):
    src = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    dst = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src* img_size
    dst = dst * np.float32((self.lane_img.shape[1], self.lane_img.shape[0]))
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (self.lane_img.shape[1], self.lane_img.shape[0]))

  def get_half_image(self, img):
    return np.sum(img[img.shape[0]//2:,:], axis=0)

  def sliding_window(self, img, draw_windows=False):
    nwindows=9 ; margin=150 ; minpix = 1
    left_a, left_b, left_c = [], [], []
    right_a, right_b, right_c = [], [], []
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    frame = np.dstack((img, img, img)) * 255
    half_image = self.get_half_image(img)
    # find peaks of left and right halves
    midpoint = int(half_image.shape[0] / 2)
    leftx_base = np.argmax(half_image[:midpoint])
    rightx_base = np.argmax(half_image[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = img.shape[0] - (window + 1) * window_height
      win_y_high = img.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      # Draw the windows on the visualization image
      if draw_windows == True:
        cv2.rectangle(frame, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3) 
        cv2.rectangle(frame, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100,255,255), 3) 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty**2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty**2 + right_fit_[1] * ploty + right_fit_[2]
    frame[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    frame[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    return (left_fitx, right_fitx)

  def get_curve(self, img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/720 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0]**2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0]**2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

  def detect_cars(self):
    # detect each car
    self.cars = self.car_classifier.detectMultiScale(self.gray)

  def detect_pedestrians(self):
    # detect each pedestrian
    self.peds = self.ped_classifier.detectMultiScale(self.gray)

  def find_lanes(self):
    self.lane_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    self.destination_image = self.hsl_detection()
    self.destination_image = self.perspective_warp_top_view()
    self.curves = self.sliding_window(self.destination_image)
    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

  def draw_cars(self):
    # place a rectangle around car
    for (x, y, w, h) in self.cars:
      self.draw_text("Car", (x+w//2, y-5), (0, 0, 255))
      self.draw_rectangle((x, y), (x+w, y+h), (0, 0, 255))

  def draw_pedestrians(self):
    # place a rectangle around pedestrians too
    for (x, y, w, h) in self.peds:
      self.draw_text("Person", (x+w//2, y-5), (0, 255, 0))
      self.draw_rectangle((x, y), (x+w, y+h), (0, 255, 0))

  def draw_line_lanes(self):
    ploty = np.linspace(0, self.lane_img.shape[0]-1, self.lane_img.shape[0])
    color_img = np.zeros_like(self.lane_img)
    left = np.array([np.transpose(np.vstack([self.curves[0], ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([self.curves[1], ploty])))])
    points = np.hstack((left, right))
    cv2.fillPoly(color_img, np.int_(points), color=[160, 32, 240])
    inv_perspective = self.inv_perspective_warp_top_view(color_img)
    self.frame = cv2.addWeighted(self.lane_img, 1, inv_perspective, 0.7, 0)

  def draw_rectangle(self, pos1, pos2, color):
    cv2.rectangle(self.frame, pos1, pos2, color, 2)

  def draw_text(self, text, pos, color):
    cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

  def draw_lanes(self, lanes, color, shade=True):
    try:
      lane_image = np.zeros_like(self.frame)
      if len(lanes) > 1:
        for line in lanes:
          x1, y1, x2, y2 = line
          cv2.polylines(lane_image, [np.array([[x1, y1], [x2, y2]])], True, color, 10)
        self.frame = cv2.addWeighted(lane_image, 0.8, self.frame, 1, 1)
        if shade:
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
        if ret and skip%1 == 0:
          if self.frame.shape[1] > 960:
            ratio = self.frame.shape[0]/self.frame.shape[1]
            # self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]//2), int(self.frame.shape[0]//2)))
            self.frame = cv2.resize(self.frame, (800, int(800*ratio)))
          """ Car and pedestrian detection """
          self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
          self.detect_cars()
          self.detect_pedestrians()
          # Depreciated way of detecting the lanes
          lanes = self.detect_lanes()
          """ Lane detection """
          self.find_lanes()
          """ DRAW EVERYTHING ON FRAME"""
          self.draw_line_lanes()
          self.draw_lanes(lanes, (255, 0, 0), shade=False)
          self.draw_cars()
          self.draw_pedestrians()
          
          """ DRAW FRAME """
          cv2.imshow('DETECTION', self.frame)
          # listen for keys
          key_pressed = cv2.waitKey(1)
      
          if key_pressed == 81 or key_pressed == 113:
            break
        else:
          break
        
        skip += 1

      except Exception as e:
        print('[DETECT] {}'.format(e))

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
