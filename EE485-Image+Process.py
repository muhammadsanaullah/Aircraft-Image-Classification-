
# coding: utf-8

# In[1]:

#importing the libraries
#import torch
import numpy as np
#import pandas as pd
#import matplotlib.image as img
#import matplotlib.pyplot as plt
from PIL import Image
#import os
#get_ipython().magic('matplotlib inline')


# In[2]:

#loading the datasets
root_path = "D:\\mbs\\BE Electrical Engineering\\Statistical Learning & Data Analytics\\Spring 2020\\Project\\data\\planes\\"
output_path = "D:\\mbs\\BE Electrical Engineering\\Statistical Learning & Data Analytics\\Spring 2020\\Project\\data\\"
drone_image_path = root_path + "drone"
fighter_image_path = root_path + "fighter-jet"
helicopter_image_path = root_path + "helicopter"
missile_image_path = root_path + "missile"
plane_image_path = root_path + "passenger-plane"
rocket_image_path = root_path + "rocket"

import glob
#take all file names in the directory intoa list for each class
drone_image_file_list = glob.glob(drone_image_path + "\\*")
fighter_image_file_list = glob.glob(fighter_image_path + "\\*")
helicopter_image_file_list = glob.glob(helicopter_image_path + "\\*")
missile_image_file_list = glob.glob(missile_image_path + "\\*")
plane_image_file_list = glob.glob(plane_image_path + "\\*")
rocket_image_file_list = glob.glob(rocket_image_path + "\\*")

#datasets summary
print(str(len(drone_image_file_list)) + " drone images found in the directory.")
print(str(len(fighter_image_file_list)) + " fighter images found in the directory.")
print(str(len(helicopter_image_file_list)) + " helicopter images found in the directory.")
print(str(len(missile_image_file_list)) + " missile images found in the directory.")
print(str(len(plane_image_file_list)) + " passenger plane images found in the directory.")
print(str(len(rocket_image_file_list)) + " rocket images found in the directory.")


# In[29]:

#processing the images from image to numpy arrays
drone_images = []
for img_file in drone_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS)) 
      drone_images.append(im)
    except:
        IOError


# In[ ]:

fighter_images = []
for img_file in fighter_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS)) 
      fighter_images.append(im)
    except:
        IOError


# In[6]:

helicopter_images = []
for img_file in helicopter_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS))
      helicopter_images.append(im)
    except:
        IOError


# In[7]:

missile_images = []
for img_file in missile_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS)) 
      missile_images.append(im)
    except:
        IOError


# In[9]:

plane_images = []
for img_file in plane_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS)) 
      plane_images.append(im)
    except:
        IOError


# In[10]:

rocket_images = []
for img_file in rocket_image_file_list:
    try:
      im = Image.open(img_file)
      im = np.asarray(im.resize((64,64), Image.ANTIALIAS)) 
      rocket_images.append(im)
    except: 
        IOError


# In[32]:

index_list= []
for i in range(len(drone_images)):
    if(drone_images[i].shape !=(64,64,3)):
        index_list.append(i)
drone_images = [drone_images[i] for i in range(len(drone_images)) if i not in index_list]


# In[33]:

index_list= []
for i in range(len(fighter_images)):
    if(fighter_images[i].shape !=(64,64,3)):
        index_list.append(i)
fighter_images = [fighter_images[i] for i in range(len(fighter_images)) if i not in index_list]


# In[34]:

index_list= []
for i in range(len(helicopter_images)):
    if(helicopter_images[i].shape !=(64,64,3)):
        index_list.append(i)
helicopter_images = [helicopter_images[i] for i in range(len(helicopter_images)) if i not in index_list]


# In[35]:

index_list= []
for i in range(len(missile_images)):
    if(missile_images[i].shape !=(64,64,3)):
        index_list.append(i)
missile_images = [missile_images[i] for i in range(len(missile_images)) if i not in index_list]


# In[37]:

index_list= []
for i in range(len(plane_images)):
    if(plane_images[i].shape !=(64,64,3)):
        index_list.append(i)
plane_images = [plane_images[i] for i in range(len(plane_images)) if i not in index_list]


# In[38]:

index_list= []
for i in range(len(rocket_images)):
    if(rocket_images[i].shape !=(64,64,3)):
        index_list.append(i)
rocket_images = [rocket_images[i] for i in range(len(rocket_images)) if i not in index_list]


# In[39]:

#save the processed image data to the directory as .npy file (numpy array file)
np.save(output_path + "drone_image_data", drone_images)
np.save(output_path + "fighter_image_data", fighter_images)
np.save(output_path + "helicopter_image_data", helicopter_images)
np.save(output_path + "missile_image_data", missile_images)
np.save(output_path + "plane_image_data", plane_images)
np.save(output_path + "rocket_image_data", rocket_images)


# In[13]:



