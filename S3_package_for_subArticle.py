from distutils import filelist
import os
import glob
import cv2
# import boto3
import random
import requests
import numpy as np
import splitfolders
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from array import array
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from tensorflow.keras import regularizers
import Grouped_articles_packages as group_list
from aws_secret import aws_access_key_id, aws_secret_access_key
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, VGG16
# s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

def data_augmentation():    
    augmentation = tf.keras.Sequential([ tf.keras.layers.RandomFlip('horizontal_and_vertical'),
                                                tf.keras.layers.RandomHeight(0.1),
                                                tf.keras.layers.RandomWidth(0.1),
                                                tf.keras.layers.RandomZoom(0.2),
                                                tf.keras.layers.RandomRotation(0.2),])
    return augmentation

def split_to_train_test_valid(mainDirName:str):
    '''
    Recieves:
    mainDirName: folder where article->images are save 

    Creates an output folder with train->article->images
                                  test->articles->images
                                  val->articles->images
    '''
    splitfolders.ratio(mainDirName, output="output", seed=1337, ratio=(.7, .25, 0.05), group_prefix=None, move=False) # default values

def split_to_train_valid(mainDirName:str):
    '''
    Recieves:
    mainDirName: folder where article->images are save 

    Creates an output folder with train->article->images
                                  valid->articles->images
    '''
    splitfolders.ratio(mainDirName, output="output", seed=1337, ratio=(.75, .25), group_prefix=None, move=False) # default values

# Preview augmented dataset
def preview_augmented_data(data, augmentation):
    for image, label in data.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

def dir_path(mainDir: str):
    getPath = os.getcwd()
    path = getPath + '\\' + mainDir
    path = path.replace('\\', '/')
    return path

def get_train_dir(path):    
    IMG_SIZE = (240, 240)
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
    IMG_SIZE = (240, 240)
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE
    validation_dir = os.path.join(path, 'val')
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)                  
    return validation_dataset

def get_test_dir(path):    
    IMG_SIZE = (240, 240)
    BATCH_SIZE = 49
    test_dir = os.path.join(path, 'test')
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    return test_dataset

def get_all_buckets():
    '''
    Returns a list of buckets in the S3
    '''
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name)

def get_all_jpg_images_path(bucket_name):
    '''
    Gets all image paths from the S3 bucket

    Returns a list of image paths
    '''
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    myImagePaths = []
    buckets = s3.Bucket(bucket_name).objects.all()
    for obj in buckets:
        if 'jpg' in obj.key:
            myImagePaths.append(obj.key)
    return myImagePaths       

def getSpecialArticleList(listOfSubclasses, imagePathsTest):
    '''
    Recieves
    -> A list of sub_classes to be retrived ['722000','722301',301, ...]
    -> A list of image paths from the S3 bucket
        The function filter out the image paths 
        corresponding to the list of sub_classes
    
    Returns a list of image paths
    '''
    # Get all sub-unique-classes from image-path corresponding to "722000"
    specialImagePath = listOfSubclasses # i.e ['722000']
    subImagePathFor722000 = []
    for i in specialImagePath:
        for j in imagePathsTest:
            if j.find(i) != -1:
                subImagePathFor722000.append(j)
    return subImagePathFor722000

def save_data_to_drive(myList, savedName):
    '''
    Recieves
    -> A list
    -> A directory name
        The function saves the list as the directory
        name in
    '''
    Classes = []
    for stringPath in myList:
        Classes.append(stringPath)

    output_file = open(savedName + '.txt', 'w')
    for _class in Classes:
        output_file.write(str(_class) + '\n')
    output_file.close()

def get_data_4rm_drive(myFile):   
    '''
    Recieves
    -> A list
        The function retrieves the list as the directory

    Returns a list
    '''   
    mylist = []
    input_file = open(myFile +'.txt', 'r')
    lines = input_file.readlines()
    for line in lines:
        mylist.append(line.strip())

    return mylist   # mylist = get_data_4rm_drive("myFile")

def all_imageClasses(myImagePaths):
    '''
    Recieves 
    -> All classes/articles list from S3 bucket
        The function extracts the classes/articles from the path,
        this means there are a lots of duplicates since there are
        more than one image belonging to a class/article 
    
    returns a list of string of all classes in S3 bucket
    '''
    ImageClasses = []
    for stringPath in myImagePaths:
        stringClass = stringPath.split('/')[0]
        ImageClasses.append(stringClass)
    return ImageClasses

def create_uniqueClasses(imageClasses):
    '''
    Recieves 
    -> A list of classes/articles from S3 bucket
        The function extracts unique classes from the list,
        removes duplicates

    Returns unique classes from classes/articles list
    '''
    uniqueClasses = []
    for element in imageClasses:
        if element not in uniqueClasses:
            uniqueClasses.append(element)
    return uniqueClasses

def image_classes_2_int(all_image_classes):
    '''
    Recieves 
    -> All classes/articles list from S3 bucket
        The function converts the list of string 
        to integer. This is important during model
        developement  
    
    returns a list of integers corresponding to input params
    '''
    # convert all generated image class from s3 bucket to integer
    # this is done cuz tensorflow only accept numbers
    a = all_image_classes
    b = create_uniqueClasses(all_image_classes)
    for i, n in enumerate(a):
        for j, x in enumerate(b):
            if n == x:
                a[i] = j
    return a

def get_all_images_in_image_size(get_all_jpg_images_path, bucket_name, image_size):
    '''
    Recieves 
    -> A list containing all paths in S3 bucket
    -> The bucket name
    -> The image size

    returns images:list from the S3 bucket
    '''

    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    myImageList = []
    for imgObj in get_all_jpg_images_path:    
        bucket = s3.Bucket(bucket_name)
        object = bucket.Object(imgObj)
        response = object.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        im = im.resize((image_size, image_size))
        myImageList.append(np.array(im))
    return myImageList

def make_augmented_images(folder_name: str, img_size:int):
    '''
    Receives
    -> The directory name
    -> The image size 
        Using the above information the function will generate
        unique datasets from the dataset in the sub_directories
        in the given directory
    -> To be used in generate_augmented_dataset()
    Returns null
    '''
    if type(folder_name) == str:
        getPath = os.getcwd()
        path = getPath + '/' + folder_name
        Imageclasses = os.listdir(path) # As cetegories
        #print(Imageclasses)
        CATEGORIES = Imageclasses

        DATADIR = path
        #print(DATADIR)
        augmentation = iaa.Sequential([
            iaa.SomeOf( 2,
            [                                 
            # Scale the Images
            iaa.Affine(scale=(0.5, 1.5)),

            # Rotate the Images
            iaa.Affine(rotate=(-45, 45)),

            # Shift the Image
            iaa.Affine(translate_percent={"x":(-0.3, 0.3),"y":(-0.3, 0.3)}),

            # Flip the Image
            iaa.Fliplr(1),

            # Increase or decrease the brightness
            iaa.Multiply((0.5, 1.5)),

            # Add Gaussian Blur
            iaa.GaussianBlur(sigma=(1.0, 3.0)),

            # Add Gaussian Noise
            iaa.AdditiveGaussianNoise(scale=(0.05*255, 0.05*255)),

            # Add large noise to image
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.10, 0.30)),

            iaa.CLAHE(clip_limit=(1, 10))
            

        ])
        ])

        IMG_SIZE = img_size
        for category in CATEGORIES:  # in dataset/DummyImages

            path = os.path.join(DATADIR, category)  # create path to DummyImages

            for img in tqdm(os.listdir(path)):  # iterate over each image per category
                img_array = cv2.imread(os.path.join(path,img) ,cv2.COLOR_BGR2RGB)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                augImg = augmentation(image = new_array)
                cv2.imwrite(os.path.join(path, 'aug' + str(img)), augImg)
    else:
        print('argument must be a string')
    return

# Fourth step
def generate_augmented_dataset(dirToBeSavedDataset:str, numOfTimes:int, img_size:int):
    '''
    Receives
    -> The directory name
    -> The number of times you want to generate dataset
    -> calls the make_augmented_images(dir, img_size)
    -> The image size 
        Using the above information the function will generate
        unique datasets from the dataset in the sub_directories
        in the given directory
    Returns null
    '''
    if numOfTimes == False:
        # We can also add some logic here for spliting ratio
        #splitfolders.ratio(dirToBeSavedDataset, output="output", seed=1337, ratio=(.7, .25, 0.05), group_prefix=None, move=False) # default values
        return 
    else:
        for i in range(numOfTimes):
            make_augmented_images(dirToBeSavedDataset, img_size)

# We can also add some logic here for spliting ratio
# splitfolders.ratio(dirToBeSavedDataset, output="output", seed=1337, ratio=(.7, .25, 0.05), group_prefix=None, move=False) # default values

# Fifth step
def create_X_y_training_data(dirToBeSavedDataset: str, img_size:int):    
    '''
    Recieves 
    -> The directory name
    -> The image size 
    This directory should contain sub_directory that contains the images
    i.e mainDir/classes/images

    Returns X:list, y:list
    '''
    getPath = os.getcwd()
    path = getPath + '/' + dirToBeSavedDataset
    Imageclasses = os.listdir(path) # As cetegories
    CATEGORIES = Imageclasses
    DATADIR = path.replace('\\', '/')
    
    IMG_SIZE = img_size
    training_data = []
    X = []
    y = []

    for category in CATEGORIES:  # in dataset/FrontBackDataset

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or 1 or 2).

        for img in tqdm(os.listdir(path)):  # iterate over each image per category
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.COLOR_BGR2RGB)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    random.shuffle(training_data)
    for features ,label in training_data:
        X.append(features)
        y.append(label)    

    #save_X_y(X, y, IMG_SIZE)
    return X, y

def combine_imagelist_labellist_generate_Xy(imageList, labelList, test_data_size:int):
    '''
    Receives
    -> image-list
    -> label-list
    -> The number of images to be kept for testing.
        Using the parameters provided a shuffle is done and dataset for training is generated
    Returns train_X, train_y, test_X, test_y
    '''
    ImageAndClassesCombined = []
    for i in range(len(labelList)):
        ImageAndClassesCombined.append([imageList[i], labelList[i]])
    random.shuffle(ImageAndClassesCombined)

    train_X, train_y, test_X, test_y = create_train_test_dataset(ImageAndClassesCombined, test_data_size)
    return train_X, train_y, test_X, test_y

# Sixth step
def create_train_test_dataset(ImageAndClassesCombined, test_data_size:int):
    '''
    Receives 
    -> The a list of image-list and label-list i.e [[image-list][label-list]]
    -> The number of images to be kept for testing
    -> To be used in combine_imagelist_labellist_generate_Xy()

    Returns train_X:list, train_y:list, test_X:list, test_y:list
    '''
    X = []
    y = []
    for i in ImageAndClassesCombined:
        X.append(i[0])
        y.append(i[1])
    train_X = np.array(X[test_data_size:])
    train_y = np.array(y[test_data_size:])
    test_X = np.array(X[:test_data_size])
    test_y = np.array(y[:test_data_size])

    return train_X, train_y, test_X, test_y

# Function for downloading image and saving it to disc 
def save_image(path2saveImage, s3ImagePath, img_size):
    bucket = s3.Bucket('winkler-images-prod-search')
    object = bucket.Object(s3ImagePath)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    im = im.resize((img_size, img_size))
    if 'jpg' == s3ImagePath.split('/')[1][-3:]:
        im.save(path2saveImage+'/'+s3ImagePath.split('/')[1], 'JPEG') 
    else:
        return

# Third step where imagePaths is from get_and_save_special_classes()
def save_image_to_drive(dirToBeSavedDataset:str, imagePaths:list, img_size:int): 
    # [{className1: [imagePaths]}, {className2: [imagePaths]}]
    # Change the imageListFor72200 to the desired listObject
    path = os.getcwd()                                              # Get the current working directory
    imageDir = path + '/' + dirToBeSavedDataset                     # Create another directory path
    if os.path.isdir(imageDir) == False:                            # Make a new direectory if directory is not present
        os.mkdir(imageDir)
    for imagePath in imagePaths:                                    # Loop through the imagePath of objects
        for key, vals in  imagePath.items():                        # Get the key and value from the object
            for val in vals:                                    
                path = os.path.join(imageDir, key)                  # Create another directory inside current path
                Path(path).mkdir(parents=True, exist_ok=True)       # make a new directory inside the current path if not there already
                save_image(path, val, img_size) 

# First step
def get_subpath_for_download(subList:list, bucketName:str):
    '''
    -> get_all_jpg_images_path: Get all paths
    -> getSpecialArticleList: Get all sub_classes
    -> all_imageClasses: Get all classes 
    -> create_uniqueClasses: Generate unique class from all_imageClasses()
    -> save_data_to_drive: Save unique classes to drive

    Returns  imagePaths and uniqueClasses
    '''
    imagePathsTest = get_all_jpg_images_path(bucketName)
    print("getting sub_path from S3 bucket ...")
    subImagePaths = getSpecialArticleList(subList, imagePathsTest)
    print("get all classes from this path ...")
    all_image_classes = all_imageClasses(subImagePaths)
    print("extracting unique classes for testing ...")
    uniqueClasses = create_uniqueClasses(all_image_classes)
    save_data_to_drive(uniqueClasses, 'uniqueClasses')
    print('unique classes length is ', len( uniqueClasses))
    return subImagePaths, uniqueClasses

def get_allpath_for_download(bucketName:str):
    '''
    -> get_all_jpg_images_path: Get all paths
    -> getSpecialArticleList: Get all sub_classes
    -> all_imageClasses: Get all classes 
    -> create_uniqueClasses: Generate unique class from all_imageClasses()
    -> save_data_to_drive: Save unique classes to drive

    Returns  imagePaths and uniqueClasses
    '''
    imagePathsTest = get_all_jpg_images_path(bucketName)
    print("get all classes from this path ...")
    all_image_classes = all_imageClasses(imagePathsTest)
    print("extracting unique classes for testing ...")
    uniqueClasses = create_uniqueClasses(all_image_classes)
    save_data_to_drive(uniqueClasses, 'uniqueClasses')
    print('unique classes length is ', len( uniqueClasses))
    return imagePathsTest, uniqueClasses

def get_and_save_special_classes(subImagePaths, uniqueClasses):
    '''
    Recieves 
    -> A list of sub_image_paths
    -> A list of unique classes
        Using the params above the function returns a special list
        i.e [{class2:[subImagePaths]}, {class2:[subImagePaths]}...]

    Returns a list [{class2:[subImagePaths]}, {class2:[subImagePaths]}...]
    '''
    # Group paths according to unique classes this will be used to save to disk
    imageListFor722000 = []
    for i in range(len(uniqueClasses)):
        imageListFor722000.append({uniqueClasses[i]: [x for x in subImagePaths if uniqueClasses[i] in x]})
    return imageListFor722000

# group the sub_paths generated using uniqueClasses
def group_subpath_with_uniqueClasses(subImagePathFor722000, uniqueClasses):
    # link classes to corresponding image path
    grouped_subPath_with_uniqueClasses = []
    for i in range(len(uniqueClasses)):
        grouped_subPath_with_uniqueClasses.append([x for x in subImagePathFor722000 if uniqueClasses[i] in x])
    return grouped_subPath_with_uniqueClasses

# Declare a funtion to filter data from the first list
def Filter(sub_path, sub_articles):
    return [n for n in sub_path if any(m in n for m in sub_articles)]

# Create a sub_class from the already grouped_subPath_with_uniqueClasses
def create_sub_sub_articles(grouped_subPath_with_uniqueClasses, sub_articles):
    myList = []
    for i in grouped_subPath_with_uniqueClasses:
        for j in range(len(sub_articles)):
            myList.append({i[0].split("/")[0]+'_V'+str(j+1): Filter(i, sub_articles[j])})
    return myList

def prepare_flow_from_dir_dataset(mainDirName):
    '''
    Recieves
    mainDirName: a folder with train->article->images
                               test->articles->images
                               valid->articles->images

    Returns tensorflow train_dataset, test_dataset, valid_dataset 
    '''
    path = dir_path(mainDirName)
    train_dataset = get_train_dir(path)
    test_dataset = get_test_dir(path)
    valid_dataset = get_validation_dir(path)

    return train_dataset, test_dataset, valid_dataset

def download_prepare_subArticles_dataset_Xy(listOfSubArticles:list, bucketName:str, imageSize:int, testDataSize:int):
    '''
    To be used for sub_Articles

    Receives
    -> listOfSubArticles: A list of sub class/es ['722000',...]
    -> bucketName: The bucket name
    -> imageSize: The image size in integer
    -> testDataSize: The number of test images to be kept for testing 
    '''
    print("get all image path from S3 bucket ...")
    imagePathsTest = get_all_jpg_images_path(bucketName)

    print("getting sub_path from S3 bucket ...")
    subImagePaths = getSpecialArticleList(listOfSubArticles, imagePathsTest)

    print("get all classes from this path ...")
    all_image_classes = all_imageClasses(subImagePaths)

    print("extracting unique classes for testing ...")
    uniqueClasses = create_uniqueClasses(all_image_classes)
    save_data_to_drive(all_image_classes, 'uniqueClasses')
    print('Number of classes', len(uniqueClasses))

    print("convert string classes to integer equivalent ...")
    labelList = image_classes_2_int(all_image_classes)

    print("get all images in the sub_path from S3 ...")
    imageList = get_all_images_in_image_size(subImagePaths, bucketName, imageSize)

    print("split imageList, labelList to get train_x, train_y, test_x, test_y ...")
    train_x, train_y, test_x, test_y = combine_imagelist_labellist_generate_Xy(imageList, labelList, testDataSize)

    return train_x, train_y, test_x, test_y

def download_prepare_sub_subArticles_Xy(sub_article:list, sub_sub_articles:list, bucket_name, dir2save_img, img_size, numOfAug:int, testImgSize):
    '''
    To be used for sub_subArticles if augmentation is needed or not on the dataset

    Recieves
    -> sub_article: A list of sub_articles
    -> sub_sub_articles: A list of sub_sub_articles
    -> The bucket name 
    -> dir2save_img: The directory where images are saved
    -> img_size: The image size i.e 240 or 400
    -> numOfAug: The number of times augmentation should be done 
            i.e if numOfAug is 0, no augmentation is done
    -> The number of test images to be kept for testing

    Returns train_X, train_y, test_X, test_y
    '''
    print("Downloading ...")
    subImagePathFor722000, uniqueClasses = get_subpath_for_download(sub_article, bucket_name)
    group_subpath = group_subpath_with_uniqueClasses(subImagePathFor722000, uniqueClasses)
    create_sub_sub_article = create_sub_sub_articles(group_subpath, sub_sub_articles)
    save_image_to_drive(dir2save_img, create_sub_sub_article, img_size)
    generate_augmented_dataset(dir2save_img, numOfAug, img_size)
    X, y = create_X_y_training_data(dir2save_img, img_size)
    train_X, train_y, test_X, test_y = combine_imagelist_labellist_generate_Xy(X, y, testImgSize)

    return train_X, train_y, test_X, test_y

def download_sub_subArticles_with_aug_to_drive(sub_article:list, sub_sub_articles:list, bucket_name, dir2save_img, img_size, numOfAug:int):
    '''
    To be used for sub_subArticles if augmentation is needed or not on the dataset

    Recieves
    -> sub_article: A list of sub_articles
    -> sub_sub_articles: A list of sub_sub_articles
    -> The bucket name 
    -> dir2save_img: The directory where images are saved
    -> img_size: The image size i.e 240 or 400
    -> numOfAug: The number of times augmentation should be done 
  
    '''
    
    print("Downloading ...")
    subImagePathFor722000, uniqueClasses = get_subpath_for_download(sub_article, bucket_name)
    group_subpath = group_subpath_with_uniqueClasses(subImagePathFor722000, uniqueClasses)
    create_sub_sub_article = create_sub_sub_articles(group_subpath, sub_sub_articles)

    print("Saving sub_sub_articles to drive ...")
    save_image_to_drive(dir2save_img, create_sub_sub_article, img_size)

    print("Generating augmented dataset using dataset from drive ...")
    generate_augmented_dataset(dir2save_img, numOfAug, img_size)

def download_subArticles_with_aug_to_drive(sublist:list, bucketName:str, dirName:str, imgSize:int, numOfAug:int):
    '''
    To be used for sub_subArticles if augmentation is needed or not on the dataset

    Recieves
    -> sublist: A list of sub_articles
    -> The bucket name 
    -> dirName: The directory where images are saved
    -> imgSize: The image size i.e 240 or 400
    -> numOfAug: The number of times augmentation should be done 
    '''
    print("Get sub_class from S3 bucket and extract unique classes ...")
    subImagelist, uniqueClasses = get_subpath_for_download(sublist, bucketName)

    print("Preparing a special list sub_class path above ...")
    imagePathList = get_and_save_special_classes(subImagelist, uniqueClasses)

    print("Saving sub_sub_articles to drive ...")
    save_image_to_drive(dirName, imagePathList, imgSize)

    print("Generating augmented dataset using dataset from drive ...")
    generate_augmented_dataset(dirName, numOfAug, imgSize)

def download_allArticles_with_aug_to_drive(bucketName:str, dirName:str, imgSize:int, numOfAug:int):
    '''
    To be used for sub_subArticles if augmentation is needed or not on the dataset

    Recieves
    -> sublist: A list of sub_articles
    -> The bucket name 
    -> dirName: The directory where images are saved
    -> imgSize: The image size i.e 240 or 400
    -> numOfAug: The number of times augmentation should be done 
    '''
    print("Get sub_class from S3 bucket and extract unique classes ...")
    subImagelist, uniqueClasses = get_allpath_for_download(bucketName)

    print("Preparing a special list sub_class path above ...")
    imagePathList = get_and_save_special_classes(subImagelist, uniqueClasses)

    print("Saving sub_sub_articles to drive ...")
    save_image_to_drive(dirName, imagePathList, imgSize)

    print("Generating augmented dataset using dataset from drive ...")
    generate_augmented_dataset(dirName, numOfAug, imgSize)

# All steps combined i.e 1--6
def download_prepare_sub_articles_with_aug(sublist:list, bucketName:str, dirName:str, imgSize:int, numOfAug:int, test_data_size):
    '''
    To be used for sub_Articles if augmentation is needed or not on the dataset
    Recieves
    -> sublist: A list of sub_list
    -> bucketName: The bucket name in string
    -> dirName: Directory to save image for further preprocessing
    -> imgSize: Image size in integer
    -> numOfAug: The number of times augmentation should be done
    -> test_data_size: The test image size to be kept

    Return train_X, train_y, test_X, test_y
    '''
    print("Get sub_class from S3 bucket and extract unique classes ...")
    subImagelist, uniqueClasses = get_subpath_for_download(sublist, bucketName)

    print("Save sub_class to drive using sub_class path above ...")
    imagePathList = get_and_save_special_classes(subImagelist, uniqueClasses)
    save_image_to_drive(dirName, imagePathList, imgSize)

    print("Make augmented dataset using image from drive")
    generate_augmented_dataset(dirName, numOfAug, imgSize)

    print("Generate a numpy Xy dataset and split data for model training")
    X, y = create_X_y_training_data(dirName, imgSize)
    train_X, train_y, test_X, test_y = combine_imagelist_labellist_generate_Xy(X, y, test_data_size)

    return train_X, train_y, test_X, test_y

def buildEfficientB0_Model_Xy(train_X:list, train_y:list, numOfClass:int, batch_size:int, epoch:int, finetune_epoch:int):
    '''
    Receives
    -> train_X: A list of training images
    -> train_y: A list of training labels
    -> numOfClass: The total number of unique classes in integer
    -> batch_size: Batch size to be used in training in integer
    -> epoch: The number of epochs i.e number of times training should run
    -> finetune_epoch: The number of fine_tuned epoch

    Returns base_model_fine, model_fine, history_fine 
    '''
    print("Extracting features using pretrained model ...")
    base_model, model, history = trainEfficientNetB0_Xy(train_X, train_y, numOfClass, batch_size, epoch)
    model.save('trained_model_features_Xy.h5')

    print("Fine tuning model for better predicting ...")
    base_model_fine, model_fine, history_fine = finetuneEfficientNetB0_Xy(base_model, model, train_X, train_y, batch_size, finetune_epoch)
    model.save('trained_model_finetuned_Xy.h5')
    
    return base_model_fine, model_fine, history_fine

def buildEfficientB0_Model(trainDataset, validationDataset, numOfClass:int, epoch:int, finetune_epoch:int):
    '''
    Receives
    -> train_X: A list of training images
    -> validationDataset: A list of training labels
    -> numOfClass: The total number of unique classes in integer
    -> batch_size: Batch size to be used in training in integer
    -> epoch: The number of epochs i.e number of times training should run
    -> finetune_epoch: The number of fine_tuned epoch

    Returns base_model_fine, model_fine, history_fine 
    '''
    print("Extracting features using pretrained model ...")
    base_model, model = trainEfficientNetB0(trainDataset, validationDataset, numOfClass, epoch)
    # create a directory to save the model if not exist
    if not os.path.exists('buildEfficientB0_Model'):
        os.makedirs('buildEfficientB0_Model')
    model.save('buildEfficientB0_Model/trained_model_features.keras')

    print("Fine tuning model for better predicting ...")
    base_model_fine, model_fine, history_fine = finetuneEfficientNetB0(base_model, model, trainDataset, validationDataset, finetune_epoch)
    model.save('buildEfficientB0_Model/trained_model_finetuned.keras')
    
    return base_model_fine, model_fine, history_fine

def buildTopK_EfficientB0_Model_Xy(train_X:list, train_y:list, numOfClass:int, batch_size:int, epoch:int, finetune_epoch:int, topk:int):
    '''
    Receives
    -> train_X: A list of training images
    -> train_y: A list of training labels
    -> numOfClass: The total number of unique classes in integer
    -> batch_size: Batch size to be used in training in integer
    -> epoch: The number of epochs i.e number of times training should run
    -> finetune_epoch: The number of fine_tuned epoch

    Returns base_model_fine, model_fine, history_fine 
    '''
    print("Extracting features using pretrained model ...")
    base_model, model, history = trainEfficientNetB0_Topk_Xy(train_X, train_y, numOfClass, batch_size, epoch, topk)
    model.save('trained_model_features_topk_Xy.h5')

    print("Fine tuning model for better predicting ...")
    base_model_fine, model_fine, history_fine = finetuneEfficientNetB0_Topk_Xy(base_model, model, train_X, train_y, batch_size, finetune_epoch, topk)
    model.save('trained_model_finetuned_topk_Xy.h5')
    
    return base_model_fine, model_fine, history_fine

def buildTopK_EfficientB0_Model(train, validation, numOfClass:int, epoch:int, finetune_epoch:int, topk:int):
    '''
    Receives
    -> train: A list of training images
    -> validation: A list of training labels
    -> numOfClass: The total number of unique classes in integer
    -> batch_size: Batch size to be used in training in integer
    -> epoch: The number of epochs i.e number of times training should run
    -> finetune_epoch: The number of fine_tuned epoch

    Returns base_model_fine, model_fine, history_fine 
    '''
    print("Extracting features using pretrained model ...")
    base_model, model= trainEfficientNetB0_Topk(train, validation, numOfClass, epoch, topk)
    model.save('trained_model_features_topk.h5')

    print("Fine tuning model for better predicting ...")
    base_model_fine, model_fine, history_fine = finetuneEfficientNetB0_Topk(base_model, model, train, validation, finetune_epoch, topk)
    model.save('trained_model_finetuned_topk.h5')
    
    return base_model_fine, model_fine, history_fine

def evaluate_and_predict_Xy(test_X, test_y, model_name, uniqueClasses, numImages):
    '''
    Receives
    -> test_X: A list of testing images
    -> test_y: A list of testing labels
    -> model_name: The model name
    -> uniqueClasses: A list of unique classes 
    -> numImages: The number of images to be used for prediction

    Returns null
    Displays an image with its predicted class and the actual class for the test dataset
    '''
    print("Evaluating the model ...")
    evaluate_model_Xy(test_X, test_y, model_name)
    print("Predicting the model using test dataset ...")
    predict_test_batch_Xy(test_X, test_y, model_name, uniqueClasses, numImages)

def evaluate_and_predict(testDir, model_name, numImages):
    '''
    Receives
    -> testDir: A list of testing images
    -> model_name: The model name
    -> numImages: The number of images to be used for prediction

    Returns null
    Displays an image with its predicted class and the actual class for the test dataset
    '''
    print("Evaluating the model ...")
    evaluate_model(testDir, model_name)
    print("Predicting the model using test dataset ...")
    predict_test_batch(testDir, model_name, numImages)

def trainEfficientNetB0_Xy(myImageList:list, imageClassesInt:list, numOfClass:int, batch_size:int, epoch:int):
    # freeze the pretrained-model
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = False
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(numOfClass, activation='softmax')

    # Build the model
    inputs = tf.keras.Input(shape=(240, 240, 3))
    #x = data_augmentation()(inputs)
    x =tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)

    #mc = tf.keras.callbacks.ModelCheckpoint('efficientnet_during_training_288_classes.h5',  mode='max', verbose=1, save_best_only=True)
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(myImageList, 
                        imageClassesInt, 
                        batch_size= batch_size, 
                        epochs= epoch,  
                        validation_split= 0.25)

    return base_model, model, history

def finetuneEfficientNetB0_Xy(base_model, model, myImageList, imageClassesInt, batch_size, epoch):
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
                    metrics=['accuracy'])
    model.summary()

    history = model.fit( myImageList, 
                        imageClassesInt, 
                        batch_size= batch_size, 
                        epochs= epoch, 
                        initial_epoch=model.history.epoch[-1],
                        validation_split= 0.25)

    return base_model, model, history

def trainEfficientNetB0_Topk_Xy(myImageList:list, imageClassesInt:list, numOfClass:int, batch_size:int, epoch:int, topk:int):
    # freeze the pretrained-model
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = False
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(numOfClass, activation='softmax')

    # Build the model
    inputs = tf.keras.Input(shape=(240, 240, 3))
    #x = data_augmentation()(inputs)
    x =tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)

    #mc = tf.keras.callbacks.ModelCheckpoint('efficientnet_during_training_288_classes.h5',  mode='max', verbose=1, save_best_only=True)
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=topk, name='top_k_categorical_accuracy')])
    history = model.fit(myImageList, 
                        imageClassesInt, 
                        batch_size= batch_size, 
                        epochs= epoch,  
                        validation_split= 0.25)

    return base_model, model, history

def finetuneEfficientNetB0_Topk_Xy(base_model, model, myImageList, imageClassesInt, batch_size, epoch, topk):
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
                    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=topk, name='top_k_categorical_accuracy')])
    model.summary()

    history = model.fit( myImageList, 
                        imageClassesInt, 
                        batch_size= batch_size, 
                        epochs= epoch, 
                        initial_epoch=model.history.epoch[-1],
                        validation_split= 0.25)

    return base_model, model, history

def trainEfficientNetB0(train_dataset, validation_dataset, classNum:int, epoch:int):
    # freeze the pretrained-model
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = False
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(classNum, activation='softmax')
    
    # Build the model
    inputs = tf.keras.Input(shape=(240, 240, 3))
    x = data_augmentation()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    initial_epochs = epoch
    model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, verbose=2)

    return base_model, model

def trainVGG16(train_dataset, validation_dataset, classNum:int, epoch:int):
    # freeze the pretrained-model
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    base_model.trainable = False
    # print(len(base_model.layers))
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(classNum, activation='softmax')
    
    # Build the model
    inputs = tf.keras.Input(shape=(240, 240, 3))
    #x = data_augmentation()(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.00001
    initial_epochs = epoch

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), 
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, verbose=2)

    return base_model, model

def finetuneVGG16(base_model, model, train_dataset, validation_dataset, epoch, finetune):

    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = finetune
    base_learning_rate = 0.00001

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    fine_tune_epochs = epoch
    total_epochs =  model.history.epoch[-1] + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=model.history.epoch[-1],
                             validation_data=validation_dataset, verbose=2)
    
    return base_model, model, history_fine

def finetuneEfficientNetB0(base_model, model, train_dataset, validation_dataset, epoch):    
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(loss='sparse_categorical_crossentropy',
                #optimizer = "adam",
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
                metrics=['accuracy'])

    fine_tune_epochs = epoch
    total_epochs =  model.history.epoch[-1] + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=model.history.epoch[-1],
                             validation_data=validation_dataset, verbose=2)
    
    return base_model, model, history_fine

def trainEfficientNetB0_Topk(train_dataset, validation_dataset, classNum:int, epoch:int, topk:int):
    # freeze the pretrained-model
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = False
    # architecture for output layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(classNum, activation='softmax')
    
    # Build the model
    inputs = tf.keras.Input(shape=(240, 240, 3))
    #x = data_augmentation()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=topk, name='top_k_categorical_accuracy')])

    initial_epochs = epoch
    model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, verbose=2)

    return base_model, model

def finetuneEfficientNetB0_Topk(base_model, model, train_dataset, validation_dataset, epoch, topk):    
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(loss='sparse_categorical_crossentropy',
                #optimizer = "adam",
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
                metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=topk, name='top_k_categorical_accuracy')])

    fine_tune_epochs = epoch
    total_epochs =  model.history.epoch[-1] + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=model.history.epoch[-1],
                             validation_data=validation_dataset, verbose=2)
    
    return base_model, model, history_fine

def evaluate_model(test_images, model_name: str):
    model = tf.keras.models.load_model(model_name)
    return model.evaluate(test_images, batch_size= 10)

def evaluate_model_Xy(test_X, test_y, model_name: str):
    model = tf.keras.models.load_model(model_name)
    return model.evaluate(test_X, test_y, batch_size= 10)

def predict_test_batch(test_images, model_name, numImages):
    model = tf.keras.models.load_model(model_name)
    
    class_names = test_images.class_names
    #print(class_names)

    plt.figure(figsize=(numImages, numImages))
    for images, labels in test_images.take(1):
        for i in range(numImages):
            img_array = tf.keras.utils.img_to_array(images[i])
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            score = tf.nn.sigmoid(predictions[0])
            
            ax = plt.subplot(int(np.sqrt(numImages)), int(np.sqrt(numImages)), i + 1)
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

def predict_test_batch_Xy(test_X, test_y, model_name, uniqueClasses, numImages):
    model = tf.keras.models.load_model(model_name)
    
    class_names = uniqueClasses
    #print(class_names)

    plt.figure(figsize=(numImages, numImages))

    for i in range(numImages):
        img_array = tf.keras.utils.img_to_array(test_X[i])
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])
        
        ax = plt.subplot(int(np.sqrt(numImages)), int(np.sqrt(numImages)), i + 1)
        plt.imshow(test_X[i])
        
        if np.argmax(score) == test_y[i]:
            color = 'blue'
        else:
            color = 'red'

        plt.title("pred: {} {:2.0f}% \n true: {}".format(class_names[np.argmax(score)],
                                100*np.max(score),
                                class_names[test_y[i]]),
                                color=color)

        plt.axis("off")

def predict_more_than_one(path, test_images, model_name):  
    class_names = test_images.class_names

    plt.figure(figsize=(20, 20))
    model = tf.keras.models.load_model(model_name)
    filelist = glob.glob(path + '/*.jpg')
    predictions = []
    for i in range(len(filelist)):
        img = tf.keras.utils.load_img(filelist[i], target_size=(240, 240))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])
        
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(img)

        plt.title("pred: {}\n Accu:{:2.0f}% \n".format(class_names[np.argmax(score)],
                                100*np.max(score),
                                ))

        plt.axis("off")

def getImgIndexRate(model:str, imgPath:str, topk:int):
    imgIndexRate = []
    top_classes = []
    #myList = []
    #retrievedClasses = []

    #classname = S3_package.get_data_4rm_drive('all_S3_classname')
    model = tf.keras.models.load_model(model)
    jpgList = glob.glob(imgPath + '/*.jpg')
    jpegList = glob.glob(imgPath + '/*.jpeg')
    if 'jpg' or 'jpeg' in jpegList[0] or jpgList[0]:
        for filename in jpgList or jpegList: #assuming gif
            im = Image.open(filename)
            img = im.resize((240, 240))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = np.vstack([img_array])
            predictions = model.predict(img_array)

            score = tf.nn.sigmoid(predictions[0])
            top_values = 100 * np.array([score[i] for i in np.argsort(score)[-topk:]]) #index represents number of top_K
            top_values_index = sorted(range(len(score)), key=lambda i: score[i])[-topk:] #index represents number of top_K
            imgIndexRate.append([img, top_values_index, top_values])

    return imgIndexRate

def getIndex_classNames(model:str, imgPath:str, topk:int):
    # get the path/directory
    folder_dir = imgPath
    imageClasses = []
    imgIndexRate = []
    model = tf.keras.models.load_model(model)
    # iterate over files in
    # that directory
    for images in glob.iglob(f'{folder_dir}/*'): 
        # check if the image ends with png
        if (images.endswith(".jpg") or images.endswith(".jpeg")):
            im = Image.open(images)
            img = im.resize((240, 240))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            #img = np.vstack([img_array])

            predictions = model.predict(img_array)

            score = tf.nn.sigmoid(predictions[0])
            top_values = 100 * np.array([score[i] for i in np.argsort(score)[-topk:]]) #index represents number of top_K
            top_values_index = sorted(range(len(score)), key=lambda i: score[i])[-topk:] #index represents number of top_K
            imgIndexRate.append([img, top_values_index, top_values])
            if images.endswith(".jpg"):
                imageClasses.append(images[-15:-4])
            else:
                imageClasses.append(images[-32:-21])
    return imgIndexRate, imageClasses

def predImgInDir(imgIndexRate:list, savedClass:str):
    retrievedClasseses = []
    classname = get_data_4rm_drive(savedClass)
    #plt.figure(figsize=(25, 25))
    for i in range(len(imgIndexRate)):
        #ax = plt.subplot(5, 5, i + 1)
        retrievedClasses = []
        #plt.imshow(imgIndexRate[i][0])
        for j in range(len(imgIndexRate[i][1])):
            retrievedClasses.append([classname[imgIndexRate[i][1][j]], imgIndexRate[i][2][j]])
        retrievedClasseses.append([imgIndexRate[i][0], retrievedClasses])
    return retrievedClasseses

def printPred(imgPath, model, className, topk:int):
    imgIndexRate, imageClasses = getIndex_classNames(model, imgPath, topk)
    retrievedClasseses = predImgInDir(imgIndexRate, className)

    articleGroupedList = get_data_4rm_drive('groupListAndArticle')
    # this is used to remove the string like format from x.realines(f)
    articleGroupedList = [eval(x) for x in articleGroupedList][0]

    if topk == 20:
        plt.figure(figsize=(25, 70))
    elif topk == 10:
        plt.figure(figsize=(25, 50))
    else:
        plt.figure(figsize=(25, 50))
            
    for i in range(len(retrievedClasseses)):
        ax = plt.subplot(8, 7, i + 1)
        plt.imshow(retrievedClasseses[i][0])


        flat_list = [item for sublist in retrievedClasseses[i][1] for item in sublist]
        myClass, myGroupedList = group_list.retrieveGroupFromTop_kAndArticle(retrievedClasseses[i], imageClasses[i], articleGroupedList)
        print(myClass)
        if imageClasses[i] in flat_list:
            color = 'blue'
        elif myClass[0] in myGroupedList:
            
            color = 'green'
        else:
            color = 'red'
        if topk == 20:
            plt.title("true:{}\n {}\n {}\n {}\n {}\n {} \n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {} \n {}\n {}\n {}\n {}\n {}\n {}".format(imageClasses[i], retrievedClasseses[i][1][::-1][0], 
            retrievedClasseses[i][1][::-1][1], retrievedClasseses[i][1][::-1][2], retrievedClasseses[i][1][::-1][3], retrievedClasseses[i][1][::-1][4], retrievedClasseses[i][1][::-1][5], 
            retrievedClasseses[i][1][::-1][6], retrievedClasseses[i][1][::-1][7], retrievedClasseses[i][1][::-1][8], retrievedClasseses[i][1][::-1][9], retrievedClasseses[i][1][::-1][10], 
            retrievedClasseses[i][1][::-1][11], retrievedClasseses[i][1][::-1][12], retrievedClasseses[i][1][::-1][13], retrievedClasseses[i][1][::-1][14], retrievedClasseses[i][1][::-1][15], 
            retrievedClasseses[i][1][::-1][16], retrievedClasseses[i][1][::-1][17], retrievedClasseses[i][1][::-1][18], retrievedClasseses[i][1][::-1][19]), color=color)
        elif topk == 10:
            plt.title("{}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}".format(imageClasses[i], retrievedClasseses[i][1][::-1][0], retrievedClasseses[i][1][::-1][1], retrievedClasseses[i][1][::-1][2], retrievedClasseses[i][1][::-1][3], retrievedClasseses[i][1][::-1][4], retrievedClasseses[i][1][::-1][5], 
            retrievedClasseses[i][1][::-1][6], retrievedClasseses[i][1][::-1][7], retrievedClasseses[i][1][::-1][8], retrievedClasseses[i][1][::-1][9]), color=color)
        else:
            plt.title("true:{}\n {}\n {}\n {}\n {}\n {} ".format(imageClasses[i], retrievedClasseses[i][1][::-1][0], retrievedClasseses[i][1][::-1][1], retrievedClasseses[i][1][::-1][2], retrievedClasseses[i][1][::-1][3], retrievedClasseses[i][1][::-1][4]), color=color)
            
        plt.axis("off")
    return retrievedClasseses

def showImgClasses(retrievedClasseses, imgIndex):
    imgCont = []
    for i in retrievedClasseses:
        imgCont.append(i[1])
    reImgCont = imgCont[imgIndex]
    plt.figure(figsize=(25, 25))
    for j in range(len(reImgCont)):
        ax = plt.subplot(8, 7, j + 1)
        gifImg = 'https://images.winkler.de/images/small/'+ reImgCont[j][0] +'_small.gif'
        response = requests.get(gifImg, stream=True)
        if response.status_code == 200:
            res = response.raw
            img = Image.open(res)
            plt.imshow(img)
            plt.axis("off")    

def printPredBeta(imgPath, model, className, topk):
    #data_dir = Path('testbilder-beta')
    #class_name = np.array(sorted([item.name[:-18] for item in data_dir.glob('*') if item.name != "testbilder-beta.txt"]))
    imgIndexRate, class_name = getIndex_classNames(model, imgPath, topk)
    img2Pred = predImgInDir(imgIndexRate, className)

    articleGroupedList = get_data_4rm_drive('groupListAndArticle')
    # this is used to remove the string like format from x.realines(f)
    articleGroupedList = [eval(x) for x in articleGroupedList][0]

    if topk == 20:
        plt.figure(figsize=(35, 165))
    elif topk == 10:
        plt.figure(figsize=(35, 100))
    else:
        plt.figure(figsize=(35, 75))
            
    for i in range(len(img2Pred)): 
        ax = plt.subplot(18, 10, i + 1)
        plt.imshow(img2Pred[i][0])

        flat_list = [item for sublist in img2Pred[i][1] for item in sublist]
        myClass, myGroupedList = group_list.retrieveGroupFromTop_kAndArticle(img2Pred[i], class_name[i], articleGroupedList)
        if class_name[i] in flat_list:
            color = 'blue'
        elif myClass[0] in myGroupedList:
            color = 'green'
        else:
            color = 'red'

        if topk == 20:
            plt.title("{}\n {}\n {}\n {}\n {} \n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {} \n {}\n {}\n {}\n {}\n {}\n {}\n {}".format(class_name[i], img2Pred[i][1][::-1][0], 
            img2Pred[i][1][::-1][1], img2Pred[i][1][::-1][2], img2Pred[i][1][::-1][3], img2Pred[i][1][::-1][4], img2Pred[i][1][::-1][5], 
            img2Pred[i][1][::-1][6], img2Pred[i][1][::-1][7], img2Pred[i][1][::-1][8], img2Pred[i][1][::-1][9], img2Pred[i][1][::-1][10], 
            img2Pred[i][1][::-1][11], img2Pred[i][1][::-1][12], img2Pred[i][1][::-1][13], img2Pred[i][1][::-1][14], img2Pred[i][1][::-1][15], 
            img2Pred[i][1][::-1][16], img2Pred[i][1][::-1][17], img2Pred[i][1][::-1][18], img2Pred[i][1][::-1][19]), color=color)       
        elif topk == 10:
            plt.title("{}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}".format(class_name[i], img2Pred[i][1][::-1][0], img2Pred[i][1][::-1][1], img2Pred[i][1][::-1][2], img2Pred[i][1][::-1][3], img2Pred[i][1][::-1][4], img2Pred[i][1][::-1][5], 
            img2Pred[i][1][::-1][6], img2Pred[i][1][::-1][7], img2Pred[i][1][::-1][8], img2Pred[i][1][::-1][9]), color=color)
        else:
            plt.title("true:{}\n {}\n {}\n {}\n {}\n {} ".format(class_name[i], img2Pred[i][1][::-1][0], img2Pred[i][1][::-1][1], img2Pred[i][1][::-1][2], img2Pred[i][1][::-1][3], img2Pred[i][1][::-1][4]), color=color)
            
        plt.axis("off")
    return img2Pred