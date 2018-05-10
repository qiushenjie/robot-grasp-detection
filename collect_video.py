import cv2
import mmap
import contextlib
import time

'''
 * Collect video from USE camera
 * write image files to ramdisk
 * write image file name to mmap object, sharing memory with other program
'''

cap = cv2.VideoCapture(0)
count = 0

with contextlib.closing(mmap.mmap(-1, 1024, tagname='grasp_det', access=mmap.ACCESS_WRITE)) as m:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            msg = "Z:\\%04d.jpg" % count
            print("Sending "+msg)
            print("time:", int(round(time.time()*1000)))
            #cv2.imshow("Video", frame)
            cv2.imwrite("Z:\\%04d.jpg" % count, frame)
            if cv2.waitKey(20) == ord('q'):
                break
            m.seek(0)
            m.write(msg.encode('utf-8'))
            m.flush()
            count += 1

            if count == 1000:
                count = 0
        else:
            break

    print("Release")
    cap.release()