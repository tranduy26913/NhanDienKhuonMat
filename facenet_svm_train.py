from os import listdir
import os
import numpy as np
from os.path import isdir
import cv2
import pickle
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



faces_folder ="data/faces/"
facenet_model = keras.models.load_model('models/facenet_keras.h5')
def get_embedding(img):
	# scale pixel values
	img = img.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = img.mean(), img.std()
	img = (img - mean) / std
 
	# transform face into one sample
	samples = np.expand_dims(img, axis=0)
	# make prediction to get embedding
	return facenet_model.predict(samples)[0]

def load_faces(train_folder):
    X_train = []
    y_train = []

    # enumerate folders, on per class
    for folder in listdir(train_folder):
        # Lặp qua các file trong từng thư mục chứa các ảnh khuôn mặt
        print(f'Đang load ảnh của {folder}')
        for file in listdir(train_folder + folder):
            # Read file
            image = cv2.imread(train_folder + folder + "/" + file,cv2.IMREAD_COLOR)
            image = np.asarray(image)

            # Thêm vào X
            X_train.append(image)
            y_train.append(folder)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f'X_shape:{X_train.shape}')

    # Convert du lieu y_train
    output_enc = LabelEncoder()
    output_enc.fit(y_train)
    y_train = output_enc.transform(y_train)
    pkl_filename = "output_enc.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(output_enc, file)

    # Convert du lieu X_train sang embeding
    X_train_emb = []
    for x in X_train:
        X_train_emb.append(get_embedding(x))

    X_train_emb = np.array(X_train_emb)
    
    print("Load faces done!")
    
    return X_train_emb, y_train


# Main program
X_train, y_train = load_faces(faces_folder)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train SVM với kernel tuyến tính
svc= SVC(kernel='linear',probability=True)
svc.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)

acc_svc = accuracy_score(y_test, svc.predict(X_test))
acc_knn = accuracy_score(y_test, knn.predict(X_test))
print(f'SVM accuracy = {acc_svc}')
print(f'KNN accuracy = {acc_knn}')
# Save model
pkl_filename = "faces_svm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svc, file)

print("Saved model")


