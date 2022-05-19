import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END, Button, Label, Scale, DoubleVar, IntVar
from tkinter.filedialog import Open, SaveAs
import cv2
import numpy as np
import pickle
from tensorflow import keras
import time
import cv2
import dlib


predictor = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
facenet_model = keras.models.load_model('models/facenet_keras.h5')
dest_size = (160, 160)

# Load SVM model từ file
pkl_filename = 'faces_svm.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Load ouput_enc từ file để hiển thị nhãn
pkl_filename = 'output_enc.pkl'
with open(pkl_filename, 'rb') as file:
    output_enc = pickle.load(file)


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = frame
    dets = detector(gray, 0)

    if len(dets) > 0:
        rect = dets[0]
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()

        face = pixels[y:h, x:w]
        face = cv2.resize(face, dest_size)
        # Lây face embeding
        face_emb = get_embedding(facenet_model, np.array(face))
        # Chuyển thành tensor
        face_emb = np.expand_dims(face_emb, axis=0)
        # Predict qua SVM
        y_hat = pickle_model.predict_proba(face_emb)
    
        probability = round(np.max(y_hat) *100,2)
        label = [np.argmax(y_hat)]
        # Lấy nhãn và viết lên ảnh
        predict_names = output_enc.inverse_transform(label)
        
        if predict_names != None:
            if probability > 70:
                text = predict_names[0]+f'({probability})'
            else:
                text = "Khong xac dinh"
            frame = cv2.rectangle(
                frame, (x - 20, y - 50), (w+20, h+20), (36, 255, 12), 2)
            cv2.putText(
                frame, text, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

class Main(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.balance = IntVar(value=1)
        self.initUI()

    def initUI(self):
        self.parent.title("Giao diện nhận diên khuôn mặt")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open Image", command=self.onOpen)
        fileMenu.add_command(label="Save", command=self.onSave)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)

        btn = Button(self.parent, text='Mở ảnh', bd='5',
                     command=self.onHandleImage)
        btn.place(x=50, y=50)

        btn = Button(self.parent, text='Mở Camera', bd='5',
                     command=self.onHandleCamera)
        btn.place(x=50, y=100)

# Set the position of button on the top of window.

        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)

    def onOpen(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            global imgin
            imgin = cv2.imread(fl)
            # imgin = cv2.imread(fl,);
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", imgin)

    def onSave(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = SaveAs(self, filetypes=ftypes)
        fl = dlg.show()
        if fl != '':
            cv2.imwrite(fl, imgout)

    def onHandleImage(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes=ftypes)
        fl = dlg.show()
        if fl == '':
            return
        global imgout

        frame = cv2.imread(fl)
        cv2.namedWindow("FACE", cv2.WINDOW_AUTOSIZE)

        frame = predict(frame)
        cv2.imshow("FACE", frame)
        cv2.waitKey(0)
        imgout = frame

    def onHandleCamera(self):
        cap = cv2.VideoCapture(1)
        while(True):

            ret, frame = cap.read()
            if not ret:
                break
            frame = predict(frame)

            # Display the resulting frame
            cv2.imshow('FACE', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


root = Tk()
Main(root)
root.geometry("640x480+100+100")
root.mainloop()
