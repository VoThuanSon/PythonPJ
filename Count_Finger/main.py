import time
import os
import cv2
import hand as htm

Folder_Path = "Fingers"
lst = os.listdir(Folder_Path)
lst_2 = []
for file in lst:
    img = cv2.imread(Folder_Path + "/" + file)
    lst_2.append(img)

pTime = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.85)
fingerid = [4,8,12,16,20]
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList, handType = detector.findPosition(frame,draw=False)
    fingers = []
    if len(lmList) != 0:
        # Ngón cái
        if handType == "Right":
            if lmList[fingerid[0]][1] < lmList[fingerid[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left
            if lmList[fingerid[0]][1] > lmList[fingerid[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 ngón còn lại
        for id in range(1,5):
            if lmList[fingerid[id]][2] < lmList[fingerid[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    count_fingers = fingers.count(1)
    h, w, c = lst_2[count_fingers - 1].shape
    frame[0:h, 0:w] = lst_2[count_fingers - 1]
    #FPS
    cv2.rectangle(frame,(0,200),(150,400),(0,255,0),-1)
    cv2.putText(frame,str(count_fingers),(30,390),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),5)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f"FPS: {int(fps)}",(150,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
    cv2.imshow("CountFinger",frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
