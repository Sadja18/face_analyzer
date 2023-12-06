#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install imutils
# !pip install filetype
# !pip install deepface
# !pip install matplotlib


# In[2]:

from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
import cv2
import numpy as np
import os
import filetype
from deepface import DeepFace
import matplotlib.pyplot as plt
import argparse


# In[3]:

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar

def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

def deep_face_analyzer(image_path):
    image = cv2.imread(image_path)
    color_img_new = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow( color_img)

    # this analyses the given image and gives values
    # when we use this for 1st time, it may give many errors 
    # and some google drive links to download some '.h5' and zip files, 
    # download and save them in the location where it shows that files are missing.
    prediction = DeepFace.analyze(color_img_new)

    return prediction

def view_it(image_path='download12.jpeg'):
    a = deep_face_analyzer(image_path)
    a= a[0]
    print("Race: ", a['dominant_race'])
    # print(a['dominant_emotion'])
    print("Age: ", a['age'])
    print("Natal Sex: ", a['dominant_gender'])

# get image path from user
if __name__== "__main__":
    parser = argparse.ArgumentParser(description ='Process some selfie.')
    parser.add_argument('--image', type=str, help="Path to the selfie image file")

    args = parser.parse_args()
    # print(args.image)
    if args.image is not None and isinstance(args.image, str):
        image_path = args.image
        
        # Analyze basic info like: Ethinicity, Age and gender
        view_it(image_path)

        # read image
        image = cv2.imread(image_path)

        # Resize image to a width of 250
        image = imutils.resize(image, width=250)

        # Show image
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        # plt.show()

        # Apply Skin Mask
        skin = extractSkin(image)

        plt.subplot(3, 1, 2)
        plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
        plt.title("Thresholded  Image")
        # plt.show()

        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
        dominantColors = extractDominantColor(skin, hasThresholding=True)

        # Show in the dominant color information
        print("Color Information")
        prety_print_data(dominantColors)

        # Show in the dominant color as bar
        print("Color Bar")
        colour_bar = plotColorBar(dominantColors)
        plt.subplot(3, 1, 3)
        plt.axis("off")
        plt.imshow(colour_bar)
        plt.title("Color Bar")

        plt.tight_layout()
        plt.savefig(f"Color_bar_{image_path}")


    else:
        print('enter valid file path')
    
    # the first cmd argument would the file itself

    # the second will be 

# In[4]:


# MARGIN = 10  # pixels
# ROW_SIZE = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# TEXT_COLOR = (255, 0, 0)  # red


# In[5]:


# IMAGE_FILE='profile.jpg'
# cascPath = "haarcascade_frontalface_default.xml"


# In[6]:


# # download from: 
# # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
# FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


# In[7]:


# # The model architecture
# # download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
# AGE_MODEL = 'weights/deploy_age.prototxt'


# In[8]:


# # The gender model architecture
# # https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
# GENDER_MODEL = 'weights/deploy_gender.prototxt'


# In[9]:


# # download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
# FACE_PROTO = "weights/deploy.prototxt.txt"


# In[10]:


# # The model pre-trained weights
# # download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
# AGE_PROTO = 'weights/age_net.caffemodel'


# In[11]:


# # The gender model pre-trained weights
# # https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
# GENDER_PROTO = 'weights/gender_net.caffemodel'


# In[12]:


# # Represent the 8 age classes of this CNN probability layer
# AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
#                  '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


# In[13]:


# # Represent the gender classes
# GENDER_LIST = ['Male', 'Female']


# In[14]:


# # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# # substraction to eliminate the effect of illunination changes
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


# In[15]:


# # Initialize frame size
# frame_width = 128
# frame_height = 170
# # load face Caffe model
# face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# # Load age prediction model
# age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# # Load gender prediction model
# gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)


# In[16]:


# def get_faces(frame, confidence_threshold=0.5):
#     """Returns the box coordinates of all detected faces"""
#     # convert the frame into a blob to be ready for NN input
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
#     # set the image as input to the NN
#     face_net.setInput(blob)
#     # perform inference and get predictions
#     output = np.squeeze(face_net.forward())
#     # initialize the result list
#     faces = []
#     # Loop over the faces detected
#     for i in range(output.shape[0]):
#         confidence = output[i, 2]
#         if confidence > confidence_threshold:
#             box = output[i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
#             # convert to integers
#             start_x, start_y, end_x, end_y = box.astype(int)
#             # widen the box a little
#             start_x, start_y, end_x, end_y = start_x - \
#                 10, start_y - 10, end_x + 10, end_y + 10
#             start_x = 0 if start_x < 0 else start_x
#             start_y = 0 if start_y < 0 else start_y
#             end_x = 0 if end_x < 0 else end_x
#             end_y = 0 if end_y < 0 else end_y
#             # append to our list
#             faces.append((start_x, start_y, end_x, end_y))
#     return faces


# In[17]:


# def cv2_imshow(title, img):
#     """Displays an image on screen and maintains the output until the user presses a key"""
#     # Display Image on screen
#     cv2.imshow(title, img)
#     # Mantain output until user presses a key
#     cv2.waitKey(0)
#     # Destroy windows when user presses a key
#     cv2.destroyAllWindows()


# In[18]:


# def get_optimal_font_scale(text, width):
#     """Determine the optimal font scale based on the hosting frame width"""
#     for scale in reversed(range(0, 60, 1)):
#         textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN , fontScale=scale/10, thickness=1)
#         new_width = textSize[0][0]
#         if (new_width <= width):
#             # print(scale/10)
#             return scale/10
#         # print(new_width)
#     return 1


# In[19]:


# # from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]
#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image
#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)
#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))
#     # resize the image
#     return cv2.resize(image, dim, interpolation = inter)


# In[20]:


# def predict_age(input_path: str):
#     """Predict the age of the faces showing in the image"""
#     # Read Input Image
#     img = cv2.imread(input_path)
#     # Take a copy of the initial image and resize it
#     frame = img.copy()
#     if frame.shape[1] > frame_width:
#         frame = image_resize(frame, width=frame_width)
#     faces = get_faces(frame)
#     for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
#         face_img = frame[start_y: end_y, start_x: end_x]
#         # image --> Input image to preprocess before passing it through our dnn for classification.
#         blob = cv2.dnn.blobFromImage(
#             image=face_img, scalefactor=1.0, size=(227, 227), 
#             mean=MODEL_MEAN_VALUES, swapRB=False
#         )
#         # Predict Age
#         age_net.setInput(blob)
#         age_preds = age_net.forward()
#         print("="*30, f"Face {i+1} Prediction Probabilities", "="*30)
#         for i in range(age_preds[0].shape[0]):
#             print(f"{AGE_INTERVALS[i]}: {age_preds[0, i]*100:.2f}%")
#         i = age_preds[0].argmax()
#         age = AGE_INTERVALS[i]
#         age_confidence_score = age_preds[0][i]
#         # Draw the box
#         label = f"Age:{age} - {age_confidence_score*100:.2f}%"
#         # print(label)
#         # get the position where to put the text
#         yPos = start_y - 15
#         while yPos < 15:
#             yPos += 15
#         # write the text into the frame
#         cv2.putText(frame, label, (start_x, yPos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
#         # draw the rectangle around the face
#         cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=1)
#     # Display processed image
#     cv2_imshow('Age Estimator', frame)
#     # save the image if you want
#     # cv2.imwrite("predicted_age.jpg", frame)


# In[21]:


# def predict_gender(input_path: str):
#     """Predict the gender of the faces showing in the image"""
#     # Read Input Image
#     img = cv2.imread(input_path)
#     # resize the image, uncomment if you want to resize the image
#     # img = cv2.resize(img, (frame_width, frame_height))
#     # Take a copy of the initial image and resize it
#     frame = img.copy()
#     if frame.shape[1] > frame_width:
#         frame = image_resize(frame, width=frame_width)
#     # predict the faces
#     faces = get_faces(frame)
#     # Loop over the faces detected
#     # for idx, face in enumerate(faces):
#     for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
#         face_img = frame[start_y: end_y, start_x: end_x]
#         # image --> Input image to preprocess before passing it through our dnn for classification.
#         # scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
#         # size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
#         # mean = mean substraction values to be substracted from every channel of the image.
#         # swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
#         blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
#             227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
#         # Predict Gender
#         gender_net.setInput(blob)
#         gender_preds = gender_net.forward()
#         i = gender_preds[0].argmax()
#         gender = GENDER_LIST[i]
#         gender_confidence_score = gender_preds[0][i]
#         # Draw the box
#         label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
#         # print(label)
#         yPos = start_y - 15
#         while yPos < 15:
#             yPos += 15
#         # get the font scale for this image size
#         optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
#         box_color = None
#         box_color = (0, 0, 0)
#         cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=box_color, thickness=1)
#         # Label processed image
#         cv2.putText(frame, label, (start_x, yPos),
#                     cv2.FONT_HERSHEY_PLAIN , optimal_font_scale, color=box_color, thickness=1)

#         # Display processed image
#     cv2_imshow("Gender Estimator", frame)
    


# In[22]:


# predict_age('./profile.jpg')


# In[23]:


# predict_gender('./profile.jpg')

