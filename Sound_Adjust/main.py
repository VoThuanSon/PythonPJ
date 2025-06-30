import math
import numpy as np
import cv2
import time
import hand as htm
pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.85)
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max, vol_step = volume.GetVolumeRange()

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lm_list = detector.findPosition(frame,draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1],lm_list[4][2]
        x2, y2 = lm_list[8][1],lm_list[8][2]
        cv2.circle(frame,(x1,y1),10,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(x2,y2),10,(255,0,0),cv2.FILLED)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(frame,(cx,cy),10,(255,255,0),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        vol = np.interp(length,[25,230],[vol_min,vol_max])
        volBar = np.interp(length,[25,230],[400,150])
        vol_per = np.interp(length,[25,230],[0,100])
        volume.SetMasterVolumeLevel(vol, None)
        if length < 25:
            cv2.circle(frame,(cx,cy),10,(0,255,0),cv2.FILLED)
        cv2.rectangle(frame,(50,150),(100,400),(0,255,0),2)
        cv2.rectangle(frame,(50,int(volBar)),(100,400),(0,255,0),cv2.FILLED)
        cv2.putText(frame, f"{int(vol_per) } %", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Sound_Adjust", frame)


    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()