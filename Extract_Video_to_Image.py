import cv2
video_path = r"" # to load video
img_path   = r"D:\TMP" #to save
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 1
while success:
  cv2.imwrite(img_path + "\\VIDEO_DATA\\my_image_%d.jpg" % count, image)    
  success, image = vidcap.read()
  print('Saved image ', count)
  count += 1