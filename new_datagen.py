# curl -o new_datagen.py https://transfer.sh/4BBfJ/new_datagen.py
import os
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image
from tensorflow import keras


CATEGORIES =  {
    "trainId--1" : [-1],
    "trainId-0" : [7],
    "trainId-1" : [8],
    "trainId-2" : [11],
    "trainId-3" : [12],
    "trainId-4" : [13],
    "trainId-5" : [17],
    "trainId-6" : [19],
    "trainId-7" : [20],
    "trainId-8" : [21],
    "trainId-9" : [22],
    "trainId-10" : [23],
    "trainId-11" : [24],
    "trainId-12" : [25],
    "trainId-13" : [26],
    "trainId-14" : [27],
    "trainId-15" : [28],
    "trainId-16" : [31],
    "trainId-17" : [32],
    "trainId-18" : [33],
    "trainId-19" : [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30],
}




def get_data(mode):
    """
    Function get_data(mode) accepts a string mode (train, val, test) and groups the paths of
    the images from the img_folder together, while also doing the same task with the
    images from the gtimg_folder. this function return 2 lists of these groups.

    :param mode: (type : string): "train, "val", "test"
    :return: a list of ids of images and masks
    """
    if mode == 'train' or mode == 'val' or mode == 'test':
        img_folder_path = "cityscapes/leftImg8bit"
        gtimg_folder_path = "cityscapes/gtFine"
        x_paths = []
        y_paths = []
        tmp_img_folder_path = os.path.join(img_folder_path, mode)

        # walk helps finding all files in a directory
        # saving all the images in the img_folder
        for (path, _, files) in os.walk(tmp_img_folder_path):
            for file_name in files:
                if file_name.endswith('.png'):
                    x_paths.append(os.path.join(path, file_name))
        # saving all the images in the gtimg_folder
        idx = len(tmp_img_folder_path)
        for x_path in x_paths:
            y_paths.append(gtimg_folder_path + '/{}'.format(mode)+ x_path[idx:-15] + 'gtFine_labelIds.png')
        assert len(y_paths) == len(x_paths)
        return x_paths, y_paths
    else:
        print("please choose the right mode (train, val, test)")


class DataGen(keras.utils.Sequence):
    """
    this class is made for Data generation to map all the masks to the data we are going to use
    you need to define a mode for the class ('train', 'val', 'test') to generate data
    ex:
    data_gen = DataGen(mode='train')
    """
    def __init__(self, mode, batch_size=8, image_height=1024, image_width=2048, image_depth=3, split=False, amount=1):
        self.train_ids, self.mask_train_ids = get_data('train')
        self.val_ids, self.mask_val_ids = get_data('val')
        self.test_ids, self.mask_test_ids = get_data('test')
        self.split = split
        self.amount = amount
        if self.split:
            self.train_ids, self.mask_train_ids = self.train_ids[:amount], self.mask_train_ids[:amount]
            self.val_ids, self.mask_val_ids = self.val_ids[:amount], self.mask_val_ids[amount]
            self.test_ids, self.mask_test_ids = self.test_ids[:amount], self.mask_test_ids[:amount]
        self.mode = mode
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.on_epoch_end()

    def __load__(self, id_num):
        """
         A function to load the corresponding both the image and the mask

        :param id_num: (type : int) the index of the ID in the list
        :return: 2 np.array, one for image, and one for masks
        """
        mode = self.mode
        if mode =='train':
            image_path = self.train_ids[id_num]
            mask_path = self.mask_train_ids[id_num]
        elif mode == 'val':
            image_path = self.val_ids_ids[id_num]
            mask_path = self.mask_train_ids[id_num]
        elif mode == 'test':
            image_path = self.test_ids[id_num]
            mask_path = self.mask_train_ids[id_num]
        else:
            print("mode does not exist")
            return

        _image = image.img_to_array(image.load_img(image_path, target_size=(self.image_height, self.image_width)))/255.
        _mask = image.img_to_array(image.load_img(mask_path, color_mode="grayscale", target_size=(self.image_height, self.image_width)))
        mask = np.zeros((_mask.shape[0], _mask.shape[1], 21))
        for i in range(-1, 34):
            if i in CATEGORIES['trainId--1']:
                mask[:,:,0] = np.logical_or(mask[:,:,0],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-0']:
                mask[:,:,1] = np.logical_or(mask[:,:,1],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-1']:
                mask[:,:,2] = np.logical_or(mask[:,:,2],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-2']:
                mask[:,:,3] = np.logical_or(mask[:,:,3],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-3']:
                mask[:,:,4] = np.logical_or(mask[:,:,4],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-4']:
                mask[:,:,5] = np.logical_or(mask[:,:,5],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-5']:
                mask[:,:,6] = np.logical_or(mask[:,:,6],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-6']:
                mask[:,:,7] = np.logical_or(mask[:,:,7],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-7']:
                mask[:,:,7] = np.logical_or(mask[:,:,8],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-8']:
                mask[:,:,1] = np.logical_or(mask[:,:,9],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-9']:
                mask[:,:,2] = np.logical_or(mask[:,:,10],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-10']:
                mask[:,:,3] = np.logical_or(mask[:,:,11],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-11']:
                mask[:,:,4] = np.logical_or(mask[:,:,12],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-12']:
                mask[:,:,5] = np.logical_or(mask[:,:,13],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-13']:
                mask[:,:,6] = np.logical_or(mask[:,:,14],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-14']:
                mask[:,:,7] = np.logical_or(mask[:,:,15],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-15']:
                mask[:,:,7] = np.logical_or(mask[:,:,16],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-16']:
                mask[:,:,3] = np.logical_or(mask[:,:,17],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-17']:
                mask[:,:,4] = np.logical_or(mask[:,:,18],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-18']:
                mask[:,:,5] = np.logical_or(mask[:,:,19],(_mask[:,:,0]==i))
            elif i in CATEGORIES['trainId-19']:
                mask[:,:,6] = np.logical_or(mask[:,:,20],(_mask[:,:,0]==i))
        return _image, mask

    def __getitem__(self, index):
        dic = {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids
        }
        ids = dic[self.mode]
        # if (index+1)*self.batch_size > len(ids):
        #     self.batch_size = len(ids) - index*self.batch_size
        files_batch = ids[index*self.batch_size : (index+1)*self.batch_size]
        image = np.zeros((self.batch_size, self.image_height, self.image_width, 3))
        mask = np.zeros((self.batch_size, self.image_height, self.image_width, 21))
        count = 0
        for id_name in files_batch:
            # print("getting in the loop : ", count)
            _img, _mask = self.__load__(ids.index(id_name))
            image[count] = _img
            mask[count] = _mask
            count += 1
        return image, mask

    # def __getitem__(self, index):
    #     dic = {
    #         "train": self.train_ids,
    #         "val": self.val_ids,
    #         "test": self.test_ids
    #     }
    #     ids = dic[self.mode]
    #     count = 0
    #     while True:
    #         img_batch = np.zeros((self.batch_size, self.image_height, self.image_width, 3))
    #         mask_batch = np.zeros((self.batch_size, self.image_height, self.image_width, 1))
    #         for i in range(count, count+self.batch_size):
    #             img, mask = self.__load__(i)
    #             img_batch[i-count] = img
    #             mask_batch[i-count] = mask
    #         count += self.batch_size
    #         if (count >= len(ids)):
    #             count = 0
    #         yield img_batch, mask_batch


    def on_epoch_end(self):
        pass

    def __len__(self):
        mode = self.mode
        if mode =='train':
            return int(np.ceil(len(self.train_ids)/float(self.batch_size)))
        elif mode == 'val':
            return int(np.ceil(len(self.val_ids)/float(self.batch_size)))
        elif mode == 'test':
            return int(np.ceil(len(self.test_ids)/float(self.batch_size)))
        else:
            print("mode does not exist")






# dg = DataGen('train', batch_size=2, split=True, amount=10)
# i, m = dg.__getitem__(4)
# print(m[0][:,:,0].shape)
# print(type(m))
# cvuint8 = cv2.convertScaleAbs(m[0][:,:,0])
# cv2.imshow("hi", cvuint8)
# cv2.waitKey(0)

# for i,m in dg.__getitem__(0):
#     print(i.shape)
#     print(type(i))

#     print(m.shape)
#     print(type(m))
#     cvuint8 = cv2.convertScaleAbs(m[0])
#     cv2.imshow("hi", cvuint8)
#     cv2.waitKey(0)


# print(i.shape)
# print(m.shape)
# print(np.unique(m[1]))
# cvuint8 = cv2.convertScaleAbs(m[0])
# cv2.imshow("hi", cvuint8)
# cv2.waitKey(0)




