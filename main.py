import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img #Load and Transform images
from PIL import ImageOps #Auto constract (For masked images)

import sys

DIR_img = "images" #Name of image inputs 
DIR_mask = "masks" #Name of image outputs

img_size = (512, 512) #Set size of training images
num_classes = 4
batch_size = 2

class Data(Sequence):
    #
    #We linked sending the dataset to a sequence
    #We used this method because it is safer and more straightforward than building a generator

    #The __getitem__ function retrieves a batch of input and target images based on the given index and returns them as arrays.
    #Alternatively a generator can be used
    #

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y

def pred(input_img_paths):
    test_img_paths = input_img_paths #input
    test_mask_paths = os.path.join('masks', input_img_paths[0].split('/')[-1].split('.')[0]+'.png').split('_')

    display_size = (475,350) #display size
    i = 0 #Which element on list(test_img_paths)


    try_batch_size = len(test_img_paths) #We changed batch size to prevent errors
    
    if try_batch_size > 0:
        
        val_gen = Data(try_batch_size, img_size, test_img_paths, test_mask_paths)
        
        val_preds = model.predict(val_gen)

        mask_ = np.argmax(val_preds[i], axis=-1)
        mask_ = np.expand_dims(mask_, axis=-1)
        mask_ = ImageOps.autocontrast(array_to_img(mask_))
        mask_ = img_to_array(mask_.resize(display_size))
        mask = np.zeros(display_size[::-1]+(3,))
        mask[:,:,0] = mask_[:,:,-1]
        mask[:,:,1] = mask_[:,:,-1]
        mask[:,:,2] = mask_[:,:,-1]

        image = array_to_img(mask)
        return image
        
    else:
        print(
    """
    There is an error.
    Can you check the content in "test_img_paths"?
    """
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to an image as a command-line argument.")
        sys.exit(1)

    model = load_model('model28ve213.h5')
    
    image_path = input_img_paths = sys.argv[1].split('_')

    image = pred(image_path)
    image.save('prediction.jpg')