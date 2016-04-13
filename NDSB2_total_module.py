# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:07:02 2016
@author: Sharath
"""
import os
import numpy as np
import dicom
import csv
from scipy.misc import imresize

from skimage.restoration import denoise_bilateral
from keras.utils.generic_utils import Progbar
from scipy import ndimage
from scipy.stats import norm

'''CNN layers from Keras'''
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K


# Define and initialize variables
directory = os.getcwd()
resize = True
img_shape = (64,64)

'''Loading train labels'''
def load_labels():
    # load the train labels and create a disctionary with patient id as a key and Systole and Diastole vols as values
    labels = np.loadtxt(os.path.join(directory, "train.csv"), delimiter=",", skiprows=1)
    label_map = {}
    for l in labels:
        label_map[int(l[0])] = [float(l[1]), float(l[2])]
    return label_map

def image_resize(img):
    '''Resize the image as per img_shape. Crop if required from the center'''
    if img.shape[0] < img.shape[1]:
        img = img.T
    short_side = min(img.shape[:2])
    '''Crop from center'''
    y = int(img.shape[0] - short_side/2)
    x = int(img.shape[1] - short_side/2)
    cropped_img = img[y:y+short_side, x:x+short_side]
    img = imresize(cropped_img, img_shape)
    
    return img

def load_images(folder):
    ''' Loads all the dicom images in the given folder'''
    path, patient_folders, _ = next(os.walk(os.path.join(directory,folder)))
    patient_folders = [int(p) for p in patient_folders ]
    #patient_folders = random.sample(patient_folders, 100)
    study_images = {}
    study_ids = set()
    images = []
    slice_images = []
    s_count = 0
    for subdir in patient_folders:
        subdir = str(subdir)
        img_path, subdirs,_ = next(os.walk(os.path.join(path, subdir)))
        if len(subdirs) == 1:
            img_path, slices,_ = next(os.walk(os.path.join(path, subdir, subdirs[0])))
        else:
            slices = subdirs

        print("loading images from folder %s of %s" %(subdir, folder))

        for s in slices:
            if "sax" in s:
                files = next(os.walk(os.path.join(img_path, s)))[2]
                img_count = len([file for file in files if file.endswith(".dcm")])
                for file in files:

                    if file.endswith(".dcm"):
                        file_path = os.path.join(img_path, s, file)
                        image = dicom.read_file(file_path)
                        try:
                            image = image.pixel_array.astype(float)
                            image /= np.max(image) #scaling
                            if resize:
                                image = image_resize(image)

                            images.append(image)
                        except ValueError as e:
                            print("error in making array from %s file of %s/%s"%(file,subdir,s))
                        
                        

                        if len(images) == img_count:
                            if img_count != 30:
                                x=0
                                try:
                                    while len(images) < 30:
                                        images.append(images[x])
                                        x += 1
                                    if len(images) > 30:
                                        images = images[0:30]
                                except IndexError:
                                    pass

                            slice_images.append(np.array(images))
                            images = []
                            

        print("%d sax foldes" %len(slice_images))
        study_images[subdir] = np.array(slice_images)
        study_ids.add(subdir)
        slice_images = []

    return list(study_ids), study_images

'''Load train and validation images. Resize and check if every slice has got 30 images'''

def train_npy():
    print("Writing npy file for train images...")
    ids, images = load_images("train")
    #ids, images = load_images()
    labels = load_labels()
    X = []
    y = []
    for pid in ids:
        study_images = images[pid]
        output = labels[int(pid)]
        
        for i in range(study_images.shape[0]):
            X.append(study_images[i,:,:,:])
            y.append(output)
    #Create data and labels arrays to save as .npy file
    
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    np.save('X_train.npy', X)
    np.save('y_train.npy', y)
    
    print("Done saving processed train images and labels")

'''Load validation images. Resize and check if every slice has got 30 images'''

def validation_npy():
    print("Writing npy file for validation images...")
    ids, images = load_images("validate")
    #labels = load_labels()
    X = []
    study_ids = []
    #y = []
    for pid in ids:
        study_images = images[pid]

        for i in range(study_images.shape[0]):
            X.append(study_images[i,:,:,:])
            study_ids.append(pid)

    #Create data and labels arrays to save as .npy file

    X = np.array(X, dtype=np.uint8)

    np.save('X_validation.npy', X)
    np.save('ids_validate.npy', study_ids)

    print("Done saving processed validation images")

#Main code
train_npy()
validation_npy()

#--------------------- Model ----------------
'''Load train.npy file - images stored as numpy array'''
def split_data(X, y, split_ratio = 0.85): 
    #X = preprocess(X)
    sam = np.random.rand(y.shape[0]) < split_ratio
    X_train = X[sam,:,:,:]
    X_test = X[~sam, :,:,:]
    y_train = y[sam,:]
    y_test = y[~sam,:]
    
    return X_train, y_train, X_test, y_test

'''random Rotation in every iteration'''
def rotation(X, angle_range):
    print("rotation augmentation")
    status = Progbar(X.shape[0])
    X_rotated = np.copy(X)
    for i in range(X.shape[0]):
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(X.shape[1]):
            X_rotated[i,j,:,:] = ndimage.rotate(X[i,j,:,:], angle, reshape=False, order=2)
        status.add(1)
    
    return X_rotated

'''Random shifting in every iteration- to make the fit tolerant with unseen test data'''
def shift_random(X, h_range, v_range):
    X_shift = np.copy(X)
    print("random shifting")
    status = Progbar(X.shape[0])
    for i in range(X.shape[0]):
        h_shift = np.random.rand() * h_range * 2 - h_range
        v_shift = np.random.rand() * v_range * 2 - v_range
        h_shift = int(h_shift * X.shape[2])
        v_shift = int(v_shift * X.shape[3])
        for j in range(X.shape[1]):
            X_shift = ndimage.shift(X[i,j,:,:], (h_shift, v_shift), order = 0)
        status.add(1)

    return X_shift

def my_rmse(y, y_hat):
    return K.sqrt(K.mean(K.square(y-y_hat), axis = -1))

def normalize(x):
    return (x-K.mean(x))/K.std(x)

def get_model():
    #input layer
    model = Sequential()
    model.add(Activation(activation= normalize, input_shape = (30,64,64)))
    
    #1st hidden layer - convolutional
    model.add(Convolution2D(64,3,3, border_mode = "same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64,3,3, border_mode= "valid"))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
    model.add(Dropout(0.25))
    
    #2nd Hidden layer - Convolutional
    model.add(Convolution2D(96,3,3, border_mode = "same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(96,3,3, border_mode= "valid"))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
    model.add(Dropout(0.25))

    #3rd Hidden layer - Convolutional
    model.add(Convolution2D(128,3,3, border_mode = "same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(128,3,3, border_mode= "valid"))
    model.add(Activation("relu"))
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
    model.add(Dropout(0.25))
    
    #final layer
    model.add(Flatten())
    model.add(Dense(1024, W_regularizer=l2(1e-3)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    #output layer
    model.add(Dense(1))
    
    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = my_rmse)
    return model

def get_crps(y, yhat):
    return np.sum(np.square(y-yhat))/len(y)

def get_cdf(x, std = 1e-08):
    x_cdf = np.zeros(x.shape[0], 600)
    for i in range(x.shape[0]):
        x_cdf[i] = norm.cdf(range(600), x[i], std)
    return x_cdf


#Get saved .npy files
X = np.load("X_train.npy")
y = np.load("y_train.npy")

X = X.astype("float32")
X = X/255
#Denoise the images - basic denoising using total variation method
status = Progbar(X.shape[0])
print("Denoising the images")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i,j,:,:] = denoise_bilateral(X[i,j,:,:], sigma_range=0.05, sigma_spatial=15)
    status.add(1)

#Model fitting for Systole and Diastole
epochs_iteration = 1
iterations = 150
batch_size = 32
crps = 1

#Split the data
X_train, y_train, X_test, y_test = split_data(X, y, 0.85)

min_systole_val_loss = float("inf")
min_diastole_val_loss = float("inf")

for i in range(iterations):
    print("----- starting iteration %d -----------" %(i+1))
    
    #Get defined CNN model
    systole_model = get_model()
    diastole_model = get_model()

    #For every iteration we will augment train data with random rotations and shifting to make our model
    #tolerant to unseen test data
    X_train_rs = rotation(X_train, angle_range=15)
    X_train_rs = shift_random(X_train_rs, 0.1, 0.1)

    #Fit the model for systole and diastole volumes
    fit_systole = systole_model.fit(X_train_rs, y_train[:,0], 
                      nb_epoch= epochs_iteration, batch_size= batch_size,
                      validation_data = (X_test, y_test[:,0]),
                      shuffle=True, verbose=1)

    fit_diastole = diastole_model.fit(X_train_rs, y_train[:,1], 
                      nb_epoch=epochs_iteration, batch_size=batch_size,
                      validation_data = (X_test, y_test[:,1]),
                      shuffle=True, verbose=1)


    #can be used as standard deviations in computing cdf for predicted volumes
    systole_tr_loss = fit_systole.history["loss"][-1]
    systole_val_loss = fit_systole.history["val_loss"][-1]
    diastole_tr_loss = fit_diastole.history["loss"][-1]
    diastole_val_loss = fit_diastole.history["val_loss"][-1]

    if i % crps == 0:
        #To validate augmentation effect
        systole_tr_pred = systole_model.predict(X_train, batch_size=batch_size, verbose=1)
        systole_val_pred = systole_model.predict(X_test, batch_size=batch_size, verbose=1)

        diastole_tr_pred = diastole_model.predict(X_train, batch_size=batch_size, verbose=1)
        diastole_val_pred = diastole_model.predict(X_test, batch_size=batch_size, verbose=1)

        '''Now get "cdf"s of actual volumes for train and test'''
        tr_cdf = get_cdf(np.concatenate((y_train[:,0], y_train[:,1])))
        val_cdf = get_cdf(np.concatenate((y_test[:,0], y_test[:,1])))

        '''Get cdf s of predicted volumes of train and test'''
        tr_pred_cdf = get_cdf(np.concatenate((systole_tr_pred, diastole_tr_pred)), np.mean(systole_tr_loss, diastole_tr_loss))
        val_pred_cdf = get_cdf(np.concatenate((systole_val_pred, diastole_val_pred)), np.mean(systole_val_loss, diastole_val_loss))

        '''Now calculate crps = continuous ranked probability score for train and test'''
        crps_train = get_crps(tr_cdf, tr_pred_cdf)
        crps_test = get_crps(val_cdf, val_pred_cdf)

        print("crps of train is %.4f on iteration %d" %(crps_train, i+1))
        print("crps of test is %.4f on iteration %d" %(crps_test, i+1))

    if systole_val_loss < min_systole_val_loss:
        #save weights
        min_systole_val_loss = systole_val_loss
        systole_model.save_weights("weights_systole.hdf5", overwrite = True)

    if diastole_val_loss < min_diastole_val_loss:
        #save weights
        min_diastole_val_loss = diastole_val_loss
        diastole_model.save_weights("weights_diastole.hdf5", overwrite = True)

    '''Save val_losses which are required as std deviations for generating submission file'''
    with open("val_losses.txt", mode = "wb") as f:
        f.write(str(min_systole_val_loss))
        f.write("\n")
        f.write(str(min_diastole_val_loss))
        print("best weights and min losses saved for iteration %d" %(i+1))


# Load validation data for prediction
X_sub = np.load("X_validation.npy")
ids = np.load("ids_validate.npy")

X_sub = X_sub.astype("float32")
X_sub = X_sub/255
X_sub_denoised = np.copy(X_sub)
#Denoise the images - basic denoising using total variation method
status = Progbar(X_sub.shape[0])
print("Denoising the images")
for i in range(X_sub.shape[0]):
    for j in range(X_sub.shape[1]):
        X_sub_denoised[i,j,:,:] = denoise_bilateral(X_sub[i,j,:,:], sigma_range=0.05, sigma_spatial=15)
    status.add(1)

#Accumulate study results - Prepare one record per patient by taking average of all the studies
def accumulate_studies(ids,cdf):   
    count = {}
    accum = {}
    size = cdf.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in count:
            count[idx] = 0
            accum[idx] = np.zeros((1, cdf.shape[1]), dtype = np.float32)
        count[idx] += 1
        accum[idx] += cdf[i,:]
    for i in count.keys():
        accum[i][:] /= count[i]
    return accum
    
#Predict using weights of the best CRPS iteration
#loading models with weights
model_systole = get_model()
model_diastole = get_model()
model_systole.load_weights("weights_systole.hdf5")
model_diastole.load_weights("weights_diastole.hdf5")

#load losses to be used as sigma values
with open("val_losses.txt", "rb") as f:
    val_loss_systole = float(f.readline())
    val_loss_diastole = float(f.readline())

#Predict on test data
batch_size = 16
pred_systole = model_systole.predict(X_sub_denoised, batch_size = batch_size, verbose = 1)
pred_diastole = model_diastole.predict(X_sub_denoised, batch_size = batch_size, verbose = 1)

#Get cdf for predictions
pred_sys_cdf = get_cdf(pred_systole, val_loss_systole)
pred_dia_cdf = get_cdf(pred_diastole, val_loss_diastole)

#Accumulate the results for each patient
sub_systole = accumulate_studies(ids, pred_sys_cdf)
sub_diastole = accumulate_studies(ids, pred_dia_cdf)

#generate submission
print("generating submission file")
fi = csv.reader(open("sample_submission_validate.csv"))
f = open("sub1.csv", "w") #open a new file in write mode
fo = csv.writer(f, lineterminator = "\n")
fo.writerow(fi.next())
for line in fi:
    idx = line[0]
    key, target = idx.split("_")
    key = int(key)
    out = [idx]
    if key in sub_systole:
        if target == "Diastole":
            out.extend(list(sub_diastole[key][0]))
        else:
            out.extend(list(sub_systole[key][0]))
    
    else:
        print("missed %s" %idx)
    fo.writerow(out)
f.close()
print("submission file successfully generated")
