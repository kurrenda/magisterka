import numpy as np
import cv2 as cv

from ohbot import ohbot
import time
from project.MediaPipeFileManager import MediaPipeFileManager

manager = MediaPipeFileManager('train')
# manager.export_images_to_csv(r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\training_data\*.png')
manager.export_images_to_csv(r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\training_data')
# ohbot.move(1,1,3)
# ohbot.wait(2)
# ohbot.move(1,4,3)
