import os
import numpy as np
import cv2
import random
################################################################
################################################################
############################ ABT ###############################
################################################################
### This is the main function that is used to create the ABT ###
### This is a simple implementation of                       ###
### Augmentation by translocation (ABT) for a instance       ###
### segmented dataset.                                       ###
### This script will only work out of the box if the dataset ###
### has the same structure as the example dataset.           ###
################################################################
################################################################
def main(num, base_dir, dir_list):
    # random shuffle shape list to variate the base image
    random.shuffle(dir_list) 
    # take the first shape of the list as base image
    image_dir_base = f'{base_dir}/{dir_list[0]}/inst_img'
    base_image, base_mask = get_base_image_and_mask(image_dir_base)
    # use the rest of the shapes as target images
    target_list = get_target_list(base_dir, dir_list[1:])
    # apply the ABT
    # take the shapes and translocate to the base image
    for target_image, target_mask in target_list:      
        abt_image, abt_mask = rand_shift_overlay(base_image, base_mask, 
                                                   target_image, target_mask)
        base_image = abt_image
        base_mask = abt_mask
    # save the ABT image and mask
    abt_img_folder = 'abt_img_train/'
    abt_msk_folder = 'abt_msk_train/'
    # creates a in form of: num-shapes (eg. 0000-SC1SC13LLSC3SC04HTTSC2N1SC4) 
    abt_img_name = create_abt_img_name(num, dir_list)
    save_abt_img(abt_img_folder, abt_msk_folder, abt_img_name, 
                 abt_image, abt_mask)


def rand_shift_overlay(base_image, base_mask, target_image, target_mask, max_attempts=1000):
    # random shifts until there's no overlap with base image
    for count in range(max_attempts):
        # calculate possible random shift for masks
        shift_y, shift_x = random_shift(target_mask)
        shifted_mask = shift_msk(target_mask, shift_y, shift_x)
        abt_image, abt_mask = create_abt_image_and_mask(base_mask,
                                                    base_image,
                                                    target_image, 
                                                    target_mask, 
                                                    shifted_mask)
        if not check_overlap(base_mask, shifted_mask):
            break
   
    return abt_image, abt_mask




################################################################
################################################################
### This are helper functions that are used in the main loop.###
### Most of them are SCP and dataset specific                ###
### and need modifications to work with other datasets.      ###
################################################################
################################################################



################################################################
### These helper functions are used in rand_shift_overlay    ###
################################################################
def random_shift(target_mask):
    # Find the positions of the mask
    mask_y, mask_x = np.where(target_mask > 0)
    # Generate random shift
    shift_y = np.random.randint(-np.min(mask_y), target_mask.shape[0] - np.max(mask_y))
    shift_x = np.random.randint(-np.min(mask_x), target_mask.shape[1] - np.max(mask_x))

    return shift_y, shift_x

def shift_msk(target_mask, shift_y, shift_x):
    y_coords, x_coords = np.nonzero(target_mask) # np.where(target_mask > 0)
    shifted_y_coords = y_coords + shift_y
    shifted_x_coords = x_coords + shift_x
    shifted_mask = np.zeros_like(target_mask)
    shifted_mask[shifted_y_coords, shifted_x_coords] = target_mask[y_coords, x_coords]
    return shifted_mask

def check_overlap(base_mask, shifted_mask):
    overlap = np.logical_and(base_mask != 0, shifted_mask != 0)
    return np.any(overlap)

def create_abt_image_and_mask(base_mask,base_image,target_image, 
                              target_mask, shifted_mask):
    # Cut out the objects from the target image
    target_objects = target_image.copy()
    target_objects[target_mask == 0] = 0
    # Create a copy of the base image to work on
    overlay_image = base_image.copy()
    # Overlay the shifted objects onto the base image
    overlay_image[shifted_mask != 0] = target_objects[target_mask != 0]
    # Create the overlay mask
    overlay_mask = np.where(shifted_mask != 0, shifted_mask, base_mask)
    return overlay_image, overlay_mask


################################################################


################################################################
### These helper functions are used in main                  ###
################################################################

def create_abt_img_name(num, dir_list):
    new_name = str(num).rjust(4, '0') # max 9999
    new_name += '-'
    for i in range(len(dir_list)):
        new_name += dir_list[i].split('_')[0].split('-')[0]
    return new_name

def save_abt_img(abt_img_folder, abt_msk_folder, abt_img_name, final_image, final_mask):
    # Save the overlay image
    if not os.path.exists(abt_img_folder):
        os.makedirs(abt_img_folder)
    cv2.imwrite(abt_img_folder + abt_img_name + '.jpg', final_image)

    if not os.path.exists(abt_msk_folder):
        os.makedirs(abt_msk_folder)
    # Save the overlay mask
    cv2.imwrite(abt_msk_folder + abt_img_name + '.png', final_mask)

def get_base_image_and_mask(image_dir_base):
    base_image_name_list = []
    for file in os.listdir(image_dir_base):
        if file.endswith('.jpg'):
            base_image_name_list.append(file)
    base_image_name = random.choice(base_image_name_list)
    base_image = cv2.imread(os.path.join(image_dir_base, base_image_name))

    base_mask_name = base_image_name.replace('.jpg', '.png')
    base_mask = cv2.imread(os.path.join(image_dir_base, base_mask_name), 
                           cv2.IMREAD_GRAYSCALE)
    return base_image, base_mask

def get_target_list(base_dir, dir_list):
    target_list = []
    for dir_name in dir_list:
        image_dir_target = f'{base_dir}/{dir_name}/inst_img'
        mask_dir_target = f'{base_dir}/{dir_name}/inst_msk'
        target_image, target_mask = get_target_image_and_mask(image_dir_target, mask_dir_target)
        target_list.append((target_image, target_mask))
    return target_list

def get_target_image_and_mask(image_dir_target, mask_dir_target):
    target_image_name_list = []
    for file in os.listdir(image_dir_target):
        if file.endswith('.jpg'):
            target_image_name_list.append(file)
    target_image_name = random.choice(target_image_name_list)
    target_image = cv2.imread(os.path.join(image_dir_target, target_image_name))

    target_mask_name_tmp = target_image_name.replace('.jpg', '.png')
    target_mask_name = []
    for file in os.listdir(mask_dir_target):
        if file.endswith(cutoff_after_x_underscore(target_mask_name_tmp)): # this is the unique id of the files name
            target_mask_name.append(file)
    if len(target_mask_name) == 0:
        print('no mask found for:' + mask_dir_target+target_image_name.split('_')[-1])
    test = random.choice(target_mask_name)
    target_mask = cv2.imread(os.path.join(mask_dir_target, test), 
                                cv2.IMREAD_GRAYSCALE)

    return target_image, target_mask

def cutoff_after_x_underscore(string, x=1):
    count = 0
    for index, char in enumerate(string):
        if char == '_':
            count += 1
        if count == x:
            return string[index:]
    return string
################################################################



if __name__ == '__main__':
    # list of shapes you want to use for the ABT
    dir_list = ['4H',
                'LL',
                'SC0',
                'SC1',
                'SC13',
                'SC2',
                'SC3',
                'SC4',
                'TT',
                'N1']
    # directory to the dataset folder (10S_raw_iabt)
    # for this example code the dataset has to have 
    # the same structure as the example dataset (10S_raw_iabt)
    base_dir = 'path-to/10S_raw_iabt'
    # number of abt images you want to create
    NUM_IMAGES = 5
    for i in range(NUM_IMAGES):
        main(i, base_dir, dir_list)