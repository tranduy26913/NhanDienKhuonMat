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

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
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
    X_train_len = len(X_train)
    for i in range(X_train_len):
        printProgressBar(i, X_train_len, prefix = 'Progress:', suffix = 'Complete', length = 50)
        X_train_emb.append(get_embedding(X_train[i]))

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


