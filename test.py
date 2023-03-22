import cv2

cap = cv2.VideoCapture('cam_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21,21), 0)
    _, frame_thresh = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV)

    cnt, _ = cv2.findContours(frame_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, max(cnt, key=cv2.contourArea), -1, (255,0,0), 2)
    if cnt:
        x,y,w,h = cv2.boundingRect(max(cnt, key=cv2.contourArea))
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 2)
        print(f'{cv2.contourArea(max(cnt, key=cv2.contourArea))/(frame.shape[0]*frame.shape[1])*100} %')

    cv2.imshow('frame', frame)
    if cv2.waitKey(0) == ord('q'):
        break
