import cv2
import os
from tqdm import tqdm
import copy
import numpy as np

class Preprocessor:
    def __init__(self, database_dir, new_database_dir):
        self.database_dir = database_dir
        self.new_database_dir = new_database_dir
        self.pad_x, self.pad_y = 50, 50

    @staticmethod
    def cropper(image):
        return image[3000:6000, :]

    def find_best_crop(self, grayscale_image, list_of_thresh=[60, 100, 150, 170, 200, 250]):
        x, y, w, h, area = 0, 0, 0, 0, 0
        for thresh_val in list_of_thresh:
            thresh = cv2.threshold(copy.deepcopy(grayscale_image), thresh_val, 255, cv2.THRESH_BINARY_INV)[1]
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            if len(areas) == 0:
                continue
            contour_of_interest = contours[np.argmax(areas)]
            x_t, y_t, w_t, h_t = cv2.boundingRect(contour_of_interest)
            if area < w_t*h_t < 3000*4000*0.5 and 1500 < x_t+w_t/2 < 2500 and 1000 < y_t+h_t/2 < 2000 and w_t >= h_t*0.7:
                x, y, w, h, area = x_t, y_t, w_t, h_t, w_t*h_t
        return x, y, w, h

    def fit_rect(self, image):
        grayscale_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.find_best_crop(grayscale_image)
        return image[max(0, y-self.pad_y):min(4000, y+h+self.pad_y), max(0, x-self.pad_x):min(6000, x+w+self.pad_x)]

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


class FixedCrop(Preprocessor):
    def __init__(self, database_dir, new_database_dir, folders, crop_size):
        Preprocessor.__init__(self, database_dir, new_database_dir)
        self.folders = folders
        self.crop_size = crop_size

    def fit_rect(self, image, crop_rect):
        w, h = crop_rect
        x, y = 4000//2-w//2, 3000//2-h//2
        return image[max(0, y - self.pad_y):min(4000, y + h + self.pad_y), max(0, x - self.pad_x):min(6000, x + w + self.pad_x)]

    def process(self, folder, crop_rect):
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
            image = self.fit_rect(image, crop_rect)
            cv2.imwrite(file_to_save, image)

    def resize(self, folder):
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
            cv2.imwrite(file_to_save, cv2.resize(image, (448, 448)))

    def preprocess(self):
        list_of_folders = os.listdir(self.database_dir)
        crop_rect = self.crop_size[0]
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(4)
        # pool.starmap(self.process, zip(list_of_folders, [crop_rect]*len(list_of_folders)))
        pool.map(self.resize, list_of_folders)
        pool.close()
        pool.join()



# Preprocessor(database_dir='../../database/IMAGO_test', new_database_dir='../../database/IMAGO_build').preprocess()
# Preprocessor(database_dir='/media/ankit/My Passport', new_database_dir='/media/ankit/My Passport/IMAGO_build').preprocess()
folders = [
    # '2017_03_17R'
    # '2017_03_20R',
    '2017_03_21R',
    # '2017_03_22R',
    # '2017_03_23R',
]
crop_rect = [
    # (3100, 1870),
    # (3100, 1971),
    # (2730, 2300),
    # (2200, 2200),
    # (2400, 2400),
    (3200, 3200)
]
FixedCrop(database_dir='/media/ankit/My Passport/IMAGO_build_cropped',
          new_database_dir='/media/ankit/My Passport/IMAGO_build_resized',
          folders=folders, crop_size=crop_rect).preprocess()
