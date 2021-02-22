import cv2,time
import pandas
from datetime import datetime
video = cv2.VideoCapture(0)
first_frame = None
df = pandas.DataFrame(columns = ["Start","End"])
status_list = [None,None]
time_list = []


while True:
    check,frame = video.read()
    #check is a boolean valuenot necessery
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0) # width height and blur factor
    status = 0
    if(first_frame is None):
        first_frame = gray # to capture first frame
        continue
    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    #to set a color at final frame when the diff value is more than 30 and 255 means white
    #if less than 30 no difference
    thresh_delta = cv2.dilate(thresh_delta,None,iterations = 2) # to smooth the image eleiminating the unwanted things except the object
    (cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #it is used to find the distinct objects appeared on frame  
    for contour in cnts: #captures counters which are larger than 1000 pixels
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    if(status_list[-1]==1 and status_list[-2] == 0):
        time_list.append(datetime.now())
    if(status_list[-1]==0 and status_list[-2] == 1):
        time_list.append(datetime.now())      
    cv2.imshow("delta frame",delta_frame)
    cv2.imshow("threshold delta",thresh_delta)
    cv2.imshow("color_frame",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    
for i in range(0,len(time_list),2):
    df = df.append({"Start":time_list[i], "End":time_list[i+1]},ignore_index=True)
df.to_csv("Times.csv")
print(status_list)
print(time_list)
print(df)
video.release()
cv2.destroyAllWindows
