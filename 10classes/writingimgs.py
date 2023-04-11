import numpy as np
import cv2
import os
import glob
import dlib
from tqdm import tqdm

trainPath = "/Users/kc/Desktop/acad/1q4/cs615/project/small/train/mindy_kaling/"
testPath = "/Users/kc/Desktop/acad/1q4/cs615/project/small/val/mindy_kaling/"
path = "/Users/kc/Desktop/acad/1q4/cs615/project copy 5/20classpins/"

d = 1

labels = glob.glob(path+"*/", recursive = True)
for label in labels:
    images = sorted(glob.glob(label+'*.jpg'))
    os.mkdir(label+'resizedgray')

    for i in tqdm(range(len(images))):
        img = cv2.imread(images[i])
        resized = cv2.resize(img, (35,35), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filename = label + 'resizedgray/' +  f'{d:03d}.jpg'
        cv2.imwrite(filename,gray)
        d+=1
        if d>100:
            break
    d=0

#calibrate camera
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}



cv2.destroyAllWindows()


# trainPath = "/Users/kc/Desktop/acad/1q4/cs615/project/small/train/madonna/"
# testPath = "/Users/kc/Desktop/acad/1q4/cs615/project/small/val/madonna/"
# path = trainPath
# d = 0
# images = glob.glob(path+'*.jpg')
# os.mkdir(path+'resized')

# for fname in images:
#     img = cv2.imread(fname)
#     detector = dlib.get_frontal_face_detector()
#     dimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     rects = detector(dimg)
#     for bbox in rects:
#         x1 = bbox.left()
#         y1 = bbox.top()
#         x2 = bbox.right()
#         y2 = bbox.bottom()
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#         faces = img[y1:y2, x1:x2]
#     #resized = cv2.resize(faces, (30,40), interpolation = cv2.INTER_AREA)
#     filename = path + 'resized/' + '%d.jpg'%d
#     cv2.imwrite(filename,img)
#     d+=1

# #calibrate camera
# #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# #data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}



# cv2.destroyAllWindows()

