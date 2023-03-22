import cv2

cap = cv2.VideoCapture('cam_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV)

    cnt, _ = cv2.findContours(frame_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, max(cnt, key=cv2.contourArea), -1, (255,0,0))
    if cnt:
        print(f'{cv2.contourArea(max(cnt, key=cv2.contourArea))/(frame.shape[0]*frame.shape[1])*100} %')

    cv2.imshow('frame', frame)
    if cv2.waitKey(3) == ord('q'):
        break
