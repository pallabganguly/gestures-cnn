import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1
while count != 1001:
    ret, frame = cap.read()
    cv2.rectangle(frame, (400,400), (200,200), (0,255,0),0)
    crop_img = frame[200:400, 200:400]
    value = (33, 33)
    hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # mask = cv2.inRange(hsv, np.array([0, 5, 3]), np.array([60, 255, 255]))
    # gaussian = cv2.GaussianBlur(mask, (11,11), 0)
    erosion = cv2.erode(res, None, iterations = 1)
    dilated = cv2.dilate(erosion,None,iterations = 1)
    median = cv2.medianBlur(dilated, 7)
    # median = cv2.medianBlur(dilated, 7)
    cv2.putText(frame, "Keep hand in box", (200,200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('cropped', frame)
    cv2.imshow('mask', median)
    # #
    # write_img = cv2.resize(median, (50,50))
    # cv2.imwrite('images_data/peace/'+str(count)+'.jpg',write_img)
    # print count
    # count += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
