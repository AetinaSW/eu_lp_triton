import cv2
import os

for i in os.listdir("./"):
    if i.split('.')[-1] == 'png':
        img = cv2.imread(i)
        # print(img.shape)
        img_resize = cv2.resize(img, (275,183))
        cv2.imwrite(i, img_resize)
    
    # print(img_resize.shape)