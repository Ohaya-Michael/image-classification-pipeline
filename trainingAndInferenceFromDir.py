import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, EfficientNetB0, EfficientNetB1, EfficientNetB2, MobileNet, DenseNet121 
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.metrics import categorical_crossentropy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageOps, ImageColor
import itertools
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def dir_path(mainDir: str):
    getPath = os.getcwd()
    path = getPath + '\\' + mainDir
    path = path.replace('\\', '/')
    return path

def get_train_dir(path):    
    IMG_SIZE = (400, 400)
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE
    train_dir = os.path.join(path, 'train')
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset

def get_validation_dir(path):   
    IMG_SIZE = (400, 400)
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE
    validation_dir = os.path.join(path, 'valid')
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
                         
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return validation_dataset

def get_test_dir(path):    
    IMG_SIZE = (400, 400)
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE
    test_dir = os.path.join(path, 'test')
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    
    #test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return test_dataset


def get_class_names(path: str):
    #test_dir = os.path.join(path + '/test')    
    class_name = path.class_names
    return class_name


# Preview dataset 
def preview_dataset(test_dir):
    class_names = test_dir.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in test_dir.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

            
# Preview augmented dataset
def preview_augmented_data(test_dir):
    for image, _ in test_dir.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation()(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
            
            
# Add performance to dataset
def data_augmentation():    
    augmentation = tf.keras.Sequential([ tf.keras.layers.RandomFlip('horizontal_and_vertical'),
                                                tf.keras.layers.RandomHeight(0.1),
                                                tf.keras.layers.RandomWidth(0.1),
                                                tf.keras.layers.RandomZoom(0.1),
                                                tf.keras.layers.RandomRotation(0.1),])
    return augmentation

'''
this function is not used at the moment.
instead I am using the individual params here 
'''
def preprocess_input(typ):
    if typ == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    elif typ == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    elif typ == "mobilenet":
        return tf.keras.applications.mobilenet.preprocess_input
    elif typ == "densenet":
        return tf.keras.applications.densenet.preprocess_input
    elif typ == "resnet":
        return tf.keras.applications.resnet.preprocess_input


def train_EfficientNetB0_model(train_dataset, validation_dataset, epoch:int, k: int=3):
    # freeze the pretrained-model
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = False
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(79, activation='softmax')
    
    # Build the model
    inputs = tf.keras.Input(shape=(400, 400, 3))
    x = data_augmentation()(inputs)
    x =tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)])
    initial_epochs = epoch
    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, verbose=0)

    return base_model, model


def fine_tuned_EfficientNetB0(base_model, model, train_dataset, validation_dataset, epoch, k: int=3):    
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)])

    fine_tune_epochs = epoch
    total_epochs =  model.history.epoch[-1] + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=model.history.epoch[-1],
                             validation_data=validation_dataset, verbose=0)
    
    return model, history_fine


def evaluate_model(test_images, model_name: str):
    model = tf.keras.models.load_model(model_name)
    return model.evaluate(test_images, batch_size= 10)


def save_model(model, name: str):
    model.save(name)
    
# should be used here (predict_single_image) but gives error
def my_top_n_list(n, m):
    my_list = []
    for i in range(len(n)):
        my_list.append((n[i], m[i]))
    return my_list

def predict_test_images(test_images, model_name):
    model = tf.keras.models.load_model(model_name)

    # Test all images in test set
    class_names = test_images.class_names
    predictions = model.predict(test_images)
    
    for i in range(len(predictions)):
        class_pred = class_names[np.argmax(predictions[i])]
        score = tf.nn.sigmoid(predictions[i])
        print(class_pred, "{:.3f}%".format(100 * np.max(score)))
        

def predict_test_batch(test_images, model_name):
    model = tf.keras.models.load_model(model_name)
    
    class_names = test_images.class_names

    plt.figure(figsize=(16, 16))
    for images, labels in test_images.take(1):
        for i in range(16):
            img_array = tf.keras.utils.img_to_array(images[i])
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            score = tf.nn.sigmoid(predictions[0])
            
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            if np.argmax(score) == labels[i]:
                color = 'blue'
            else:
                color = 'red'

            plt.title("pred: {} {:2.0f}% \n true: {}".format(class_names[np.argmax(score)],
                                    100*np.max(score),
                                    class_names[labels[i]]),
                                    color=color)

            plt.axis("off")
            
            
def predict_more_than_one(path, test_images, model_name):    
    
    model = tf.keras.models.load_model(model_name)
    filelist = glob.glob('{}*.JPG'.format(path))
    predictions = []
    print(filelist) 
    for i in filelist:
        img = tf.keras.utils.load_img(i, target_size=(400, 400))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        
        #class_pred = test_images.class_names[np.argmax(predictions[j])]
        score = tf.nn.sigmoid(predictions[0])
        plt.title(test_images.class_names[np.argmax(score)])
        plt.imshow(img)
        plt.axis('off')
        

def predict_single_image(test_images, path, model_name): # there should be an option for inputing top_k
    top_3_classes = []
    my_list = []
    model = tf.keras.models.load_model(model_name)

    class_names = test_images.class_names
    
    img = tf.keras.utils.load_img(path, target_size=(400, 400))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    score = tf.nn.sigmoid(predictions[0])
    top_values = 100 * np.array([score[i] for i in np.argsort(score)[-3:]]) #index represents number of top_K
    top_3_classes = []
    top_values_index = sorted(range(len(score)), key=lambda i: score[i])[-3:] #index represents number of top_K
    
    for i in range(len(top_values_index)):
        top_3_classes.append(class_names[top_values_index[i]])
        
    for i in range(len(top_values)):
        my_list.append((top_3_classes[i], top_values[i]))
        
    plt.title("pred: {} \n {}".format(class_names[np.argmax(score)], my_list))
    myplot = plt.imshow(img)
    

def get_history(pretrained_model):
    history = pretrained_model.history
    return history