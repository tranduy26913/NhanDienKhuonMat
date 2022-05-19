import cv2
import time
import os
import numpy as np
import dlib
from os.path import exists
#facenet_model = keras.models.load_model('facenet_keras.h5')
#detector = MTCNN()
image_folder = "data/images/"
face_folder = "data/faces/"
test_folder = "data/test/"

predictor = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
dest_size = (160,160)  # kích thước ảnh để train


def main():
    if not exists(image_folder):
        os.mkdir(image_folder)
    if not exists(face_folder):
        os.mkdir(face_folder)
    if not exists(test_folder):
        os.mkdir(test_folder)
    namelist = []
    infile = open('name.txt', 'r')
    for line in infile:
        namelist.append(line.strip())
    infile.close()
    for name in namelist:
        
        SplitImage(name)


def SplitImage(name):
    if exists('VID/'+name+'.MOV'):
        cap = cv2.VideoCapture('VID/'+name+'.MOV')
    elif exists('VID/'+name+'.mp4'):
        cap = cv2.VideoCapture('VID/'+name+'.mp4')
    else:
        return

    time.sleep(1)
    if cap is None or not cap.isOpened():
        print('Khong the mo file video')
        return

    imgs = []
    os.mkdir(image_folder + name)
    os.mkdir(face_folder + name)
    
    while True:
        [success, img] = cap.read()
        ch = cv2.waitKey(30)
        img = np.array(img)
        if success:
            if img.shape[0] > 800:
                img = cv2.resize(img, None, fx=0.5, fy=0.5)
            imgs.append(img)
        else:
            break
    imgs = np.array(imgs)
    # trộn ngẫu nhiên hình ảnh
    randomize = np.arange(len(imgs))
    np.random.shuffle(randomize)
    imgs = imgs[randomize]
    print(f'Đang xử lý ảnh của {name}')
    i = 0
    dem = 0
    while dem < 100 and i < len(imgs):  # 150 ảnh train
        image = imgs[i]
        filename_img = image_folder+'%s/%s_%04d.jpg' % (name, name, i+100)
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)
        if len(faces) > 0:
            
            # Chỉ lấy khuôn mặt đầu tiên, ta coi các ảnh train chỉ có 1 mặt
            rect = faces[0]
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            face = image[y:h, x:w]
            
            try:
                face = cv2.resize(face, dest_size)
                filename_face = face_folder+'%s/%s_%04d.jpg' % (name, name, i+100)
                cv2.imwrite(filename_img, image)
                cv2.imwrite(filename_face, face)
                dem = dem + 1
            except:
                i = i+1
                continue
        i = i+1
    os.mkdir(test_folder + name)
    for dem in range(15):  # ảnh test
        if i == len(imgs):
            break
        image = imgs[i]
        filename_test = test_folder+'%s/%s_%04d.jpg' % (name, name, i+100)
        cv2.imwrite(filename_test, image)
        i = i+1

    return


if __name__ == "__main__":
    main()
