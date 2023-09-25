# ABT
# This is a simple implementation of Augmentation by translocation (ABT) for a instance segmented dataset. 
# This script will only work out of the box is the dataset has the same structure as the example dataset.
import os
import numpy as np
import cv2
import random

def make_abt_images(num, base_dir, dir_list):
    random.shuffle(dir_list) 
    image_dir_base = f'{base_dir}/{dir_list[0]}/inst_img'
    mask_dir_base = f'{base_dir}/{dir_list[0]}/inst_img'

    base_image_name_list = []
    for file in os.listdir(image_dir_base):
        if file.endswith('.jpg'):
            base_image_name_list.append(file)
    base_image_name = random.choice(base_image_name_list)
    base_mask_name = base_image_name.replace('.jpg', '.png')
    final_image, final_mask = apply_rand_shift_overlay(base_dir,dir_list, base_image_name,
                                                          base_mask_name,image_dir_base)

    new_name = str(num).rjust(4, '0') # max 9999
    for i in range(len(dir_list)):
        new_name += dir_list[i].split('_')[0].split('-')[0]
    # Save the overlay image
    abt_img_folder = 'abt_img_train/'
    if not os.path.exists(abt_img_folder):
        os.makedirs(abt_img_folder)
    cv2.imwrite(abt_img_folder + new_name + '.jpg', final_image)
    abt_msk_folder = 'abt_msk_train/'
    if not os.path.exists(abt_msk_folder):
        os.makedirs(abt_msk_folder)
    # Save the overlay mask
    cv2.imwrite(abt_msk_folder + new_name + '.png', final_mask)

def apply_rand_shift_overlay(base_dir, dir_list, base_image_name, 
                             base_mask_name,image_dir_base, aug_num=1):
    base_image = cv2.imread(os.path.join(image_dir_base, base_image_name))
    base_mask = cv2.imread(os.path.join(image_dir_base, base_mask_name), 
                           cv2.IMREAD_GRAYSCALE)

    for dir_name in dir_list:
        image_dir_target = f'{base_dir}/{dir_name}/inst_img'
        mask_dir_target = f'{base_dir}/{dir_name}/inst_msk'
        target_image_name_list = []
        for file in os.listdir(image_dir_target):
            if file.endswith('.jpg'):
                target_image_name_list.append(file)
        target_image_name = random.choice(target_image_name_list)
        target_image = cv2.imread(os.path.join(image_dir_target, target_image_name))
        target_mask_name_tmp = target_image_name.replace('.jpg', '.png')
        target_mask_name = []
        for file in os.listdir(mask_dir_target):
            if file.endswith(cutoff_after_x_underscore(target_mask_name_tmp)):
                target_mask_name.append(file)
        if len(target_mask_name) == 0:
            print('no mask found for:' + mask_dir_target+target_image_name.split('_')[-1])
            continue
        test = random.choice(target_mask_name)
        target_mask = cv2.imread(os.path.join(mask_dir_target, test), 
                                 cv2.IMREAD_GRAYSCALE)
        base_image, base_mask = rand_shift_overlay(base_image, base_mask, 
                                                   target_image, target_mask)
    for j in range(aug_num):
        if bool(random.choice([True, False])):
            for dir_name in dir_list:
                image_dir_target = f'{base_dir}/{dir_name}/inst_img'
                mask_dir_target = f'{base_dir}/{dir_name}/inst_msk'
                target_image_name_list = []
                for file in os.listdir(image_dir_target):
                    if file.endswith('.jpg'):
                        target_image_name_list.append(file)
                target_image_name = random.choice(target_image_name_list)
                target_image = cv2.imread(os.path.join(image_dir_target, target_image_name))
                target_mask_name_tmp = target_image_name.replace('.jpg', '.png')
                target_mask_name = []
                for file in os.listdir(mask_dir_target):
                    if file.endswith(cutoff_after_x_underscore(target_mask_name_tmp)):
                        target_mask_name.append(file)
                if len(target_mask_name) == 0:
                    print('no mask found for:' + mask_dir_target+target_image_name.split('_')[-1])
                    continue
                test = random.choice(target_mask_name)
                target_mask = cv2.imread(os.path.join(mask_dir_target, test), 
                                        cv2.IMREAD_GRAYSCALE)
                base_image, base_mask = rand_shift_overlay(base_image, base_mask, 
                                                           target_image, target_mask)
    return base_image, base_mask

def rand_shift_overlay(base_image, base_mask, target_image, target_mask, max_attempts=1000):
    # Ensure the target image can fit into the base image
    if base_image.shape != target_image.shape:
        raise Exception('The base image and target image have different shapes.')
    # Cut out the objects from the target image
    target_objects = target_image.copy()
    target_objects[target_mask == 0] = 0
    # Create a copy of the base image to work on
    overlay_image = base_image.copy()

    # random shifts until there's no overlap with base image
    for count in range(max_attempts): 
        # Find the positions of the mask
        mask_y, mask_x = np.where(target_mask > 0)
        # Determine minimal position of the mask
        min_pos_y = np.min(mask_y) if len(mask_y) > 0 else 0
        min_pos_x = np.min(mask_x) if len(mask_x) > 0 else 0
        # Determine the maximal position of the mask
        max_pos_y = np.max(mask_y) if len(mask_y) > 0 else 0
        max_pos_x = np.max(mask_x) if len(mask_x) > 0 else 0
        # Determine the negative and positive shift in each direction
        neg_shift_y = min_pos_y 
        neg_shift_x = min_pos_x 
        # -3 to avoid edge effects can be changed to custom value
        pos_shift_y = base_mask.shape[0] - max_pos_y - 3 
        pos_shift_x = base_mask.shape[1] - max_pos_x - 3 


        # Generate random shift using the maximal shift
        shift_y = np.random.randint(-neg_shift_y, pos_shift_y)
        shift_x = np.random.randint(-neg_shift_x, pos_shift_x)

        # Create a shifted mask of the target objects
        y_coords, x_coords = np.nonzero(target_mask)
        shifted_y_coords = np.clip(y_coords + shift_y, 0, target_image.shape[0] - 1)
        shifted_x_coords = np.clip(x_coords + shift_x, 0, target_image.shape[1] - 1)
        shifted_mask = np.zeros_like(target_mask)
        shifted_mask[shifted_y_coords, shifted_x_coords] = target_mask[y_coords, x_coords]
        # Check if there's an overlap with base image objects
        overlap = np.logical_and(base_mask != 0, shifted_mask != 0)
        if not np.any(overlap):
            break # If there's no overlap, break the loop
        if count == max_attempts - 1: 
            print("Warning: Object could not be placed without overlap.")
            shifted_mask = target_mask #TODO: an other option would be to discard the object

    # Overlay the shifted objects onto the base image
    overlay_image[shifted_mask != 0] = target_objects[target_mask != 0]

    # Create the overlay mask
    overlay_mask = np.where(shifted_mask != 0, shifted_mask, base_mask)

    return overlay_image, overlay_mask



def cutoff_after_x_underscore(string, x=1):
    count = 0
    for index, char in enumerate(string):
        if char == '_':
            count += 1
        if count == x:
            return string[index:]
    return string

if __name__ == '__main__':
    NUM_IMAGES = 5
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
    base_dir = '10S_raw_abt'
    for i in range(NUM_IMAGES):
        make_abt_images(i, base_dir, dir_list)

