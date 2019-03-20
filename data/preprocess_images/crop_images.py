import cv2
import os
from tqdm import tqdm
import copy
import numpy as np

class Preprocessor:
    def __init__(self, database_dir, new_database_dir):
        self.database_dir = database_dir
        self.new_database_dir = new_database_dir
        self.pad_x, self.pad_y = 20, 20

    @staticmethod
    def cropper(image):
        return image[3000:6000, :]

    def fit_rect(self, image):
        grayscale_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(grayscale_image, 150, 255, cv2.THRESH_BINARY_INV)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(contour) for contour in contours]
        contour_of_interst = contours[np.argmax(areas)]
        x, y, w, h = cv2.boundingRect(contour_of_interst)
        return image[y-self.pad_y:y+h+self.pad_y, x-self.pad_x:x+w+self.pad_x]

    @staticmethod
    def make_dir_if_not(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def preprocess(self):
        list_of_folders = os.listdir(self.database_dir)
        for folder in tqdm(list_of_folders):
            if '2017' not in folder and '2018' not in folder and '2019' not in folder:
                continue
            current_folder = os.path.join(self.database_dir, folder)
            save_to_folder = self.make_dir_if_not(os.path.join(self.new_database_dir, folder))
            list_of_files = os.listdir(current_folder)

            for image in tqdm(list_of_files):
                file_to_read = os.path.join(current_folder, image)
                file_to_save = os.path.join(save_to_folder, image)

                if '.JPG' not in image or image.startswith('.') or os.path.exists(file_to_save):
                    continue

                image = cv2.imread(file_to_read)
                if image is None:
                    print('Failed to read an image from: {}'.format(file_to_read))
                image = self.cropper(image)
                image = self.fit_rect(image)
                cv2.imwrite(file_to_save, image)


Preprocessor(database_dir='/media/ankit/My Passport', new_database_dir='/media/ankit/My Passport/IMAGO_build').preprocess()
