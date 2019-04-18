import numpy as np
import os
import time
import pandas as pd
import copy
from sklearn.decomposition import PCA

from utils import *


class DatasetLoader():

    def __init__(self, features_dir, meta_file, test_fold=1, category_idx=None):
        # categories
        ## Animals
        category_targets = [
            ('dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects',
             'sheep', 'crow'),
            ## Natural soundscapes & water sounds
            ('rain', 'sea_waves', 'crackling_fire', 'crickets',
             'chirping_birds', 'water_drops', 'wind', 'pouring_water',
             'toilet_flush', 'thunderstorm'),
            ## Human, non-speech sounds
            ('crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
             'footsteps', 'laughing', 'brushing_teeth', 'snoring',
             'drinking_sipping'),
            ## Interior/domestic sounds
            ('door_wood_knock', 'mouse_click', 'keyboard_typing',
             'door_wood_creaks', 'can_opening', 'washing_machine',
             'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking'),
            ## Exterior/urban noises
            ('helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train',
             'church_bells', 'airplane', 'fireworks', 'hand_saw')
        ]

        self.features_dir = features_dir
        self.meta_file = meta_file

        self.df = pd.read_csv(self.meta_file)
        # esc10
        if category_idx != None:
            self.df = self.df[self.df.category.isin(
                category_targets[category_idx])]
            # re-map target integer labels
            old_target = self.df.target.unique()
            new_target = np.arange(old_target.shape[0])
            target_map = dict(zip(old_target, new_target))
            self.df.target = self.df.target.map(target_map)

        self.test_fold = test_fold
        self.classes = self.df.category.unique()

    def __load_single(self, row):
        _, fname, _ = get_path_fname_ext(row['filename'])
        feat_file = os.path.join(self.features_dir, fname + '.npy')
        x = np.load(feat_file)
        return x

    def load(self):
        self.df['feat'] = ""
        start = time.time()
        for idx, row in self.df.iterrows():
            self.df.at[idx, 'feat'] = self.__load_single(row)
        end = time.time()
        print('Time taken to load = %.2f' % (end - start))
        # split test and train
        self.df_test = self.df[self.df.fold == self.test_fold]
        self.df_train = self.df[self.df.fold != self.test_fold]
        # make copies of dataframe so we can directly write to columns without
        # pandas warning
        self.df_test = copy.deepcopy(self.df_test)
        self.df_train = copy.deepcopy(self.df_train)
        # delete original dataframe
        del self.df

        # get train and test numpy arrays
        self.x_train, self.y_train = self.__get_x_y_data(self.df_train)
        self.x_test, self.y_test = self.__get_x_y_data(self.df_test)

    def normalize(self):
        # standard normalization
        self.mean = np.mean(self.x_train, axis=0)
        self.std = np.std(self.x_train, axis=0)
        # normalize train
        self.x_train = (self.x_train - self.mean) / self.std
        # normalize test
        self.x_test = (self.x_test - self.mean) / self.std

    def apply_pca(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.x_train)
        self.x_train = self.pca.transform(self.x_train)
        self.x_test = self.pca.transform(self.x_test)

    def __get_x_y_data(self, df):
        x = np.stack(df.feat.values)
        y = df.target.values
        return x, y

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_train_data(self):
        return self.x_train, self.y_train
