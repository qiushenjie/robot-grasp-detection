import cv2
import os

if not os.path.exists('validation_img'):
    os.makedirs('validation_img')

f = open("validation_result.txt",'r')
isFilename = True
filename = ''
while True:
    line = f.readline()

    if line == '':
        break

    #print('line:', line)

    if isFilename:
        filename = line.strip()
        img = cv2.imread(filename)
        print(filename)
    else:
        edges = line[:-1].strip().split(',')
        p1 = (int(float(edges[0])/0.35), int(float(edges[1])/0.47))
        p2 = (int(float(edges[2])/0.35), int(float(edges[3])/0.47))
        p3 = (int(float(edges[4])/0.35), int(float(edges[5])/0.47))
        p4 = (int(float(edges[6])/0.35), int(float(edges[7])/0.47))

        cv2.line(img, p1, p2, (0, 0, 255))
        cv2.line(img, p2, p3, (0, 0, 255))
        cv2.line(img, p3, p4, (0, 0, 255))
        cv2.line(img, p4, p1, (0, 0, 255))

        writefile = filename.split('\\')[-1]
        cv2.imwrite('validation_img/'+writefile, img)
    isFilename = not isFilename

f.close()