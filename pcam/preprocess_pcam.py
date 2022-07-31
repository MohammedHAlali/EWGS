''' Here, the PCam dataset is resized to 224, saved in rgb, grayscale, sparsity on rgb, and sparsity on grayscale. All modes will be saved in png files'''

import os
import glob
from PIL import Image
import numpy as np

base_path = '/work/deogun/alali/data/pcam_rgb224'
out_path = '/work/deogun/alali/data/pcam_rgb224_sp'
if(not os.path.exists(out_path)):
    os.mkdir(out_path)
s_threshold = 200 #try 220 and 180
#for phase in os.listdir(base_path): #update to only process training
for phase in ['train']:
    print('starting phase: ', phase)
    phase_path = os.path.join(base_path, phase)
    out_phase_path = os.path.join(out_path, phase)
    if(not os.path.exists(out_phase_path)):
        os.mkdir(out_phase_path)
    for c in os.listdir(phase_path):
        print('class: ', c)
        class_path = os.path.join(phase_path, c)
        out_class_path = os.path.join(out_phase_path, c)
        if(not os.path.exists(out_class_path)):
            os.mkdir(out_class_path)
        filenames = os.listdir(class_path)
        print('Number of files in this class: ', len(filenames))
        for i, filename in enumerate(filenames):
            #print('filename: ', filename)
            file_path = os.path.join(class_path, filename)
            im = Image.open(file_path)
            #im = im.convert('L')
            #im = im.convert('RGB') #convert gs image to 3-channels
            #print('image size: ', im.size, ' mode: ', im.mode)
            im = im.resize(size=(224, 224))
            np_im = np.array(im)
            #print('image shape: ', np_im.shape, ' type: ', np_im.dtype)
            new_img = np.zeros(shape=(np_im.shape), dtype='uint8')
            whitish_count = 0
            for j in range(np_im.shape[0]):
                for k in range(np_im.shape[1]):
                    #check if RGB pixel color is almost white
                    if(np_im[j,k, 0] >= s_threshold and 
                            np_im[j,k, 1] >= s_threshold and 
                            np_im[j,k, 2] >= s_threshold):
                        whitish_count += 1
                    else:
                        #keep original colored pixel
                        new_img[j, k] = np_im[j, k]
            save_path = os.path.join(out_class_path, filename)
            whitish_ratio = whitish_count / (np_im.shape[0]*np_im.shape[1])
            im = Image.fromarray(new_img)
            im.save(save_path)
            if(i % 500 == 0):
                print(i, '- image shape: ', np_im.shape, ' type: ', np_im.dtype)
                print('image SAVED: ', save_path)
                print('whitish ratio: ', whitish_ratio)
        out_filenames = os.listdir(out_class_path)
        if(len(filenames) == len(out_filenames)):
            print('count is equal')
        else:
            raise ValueError('ERROR: count not equal, new filename count: ', len(out_filenames))
print('done')
