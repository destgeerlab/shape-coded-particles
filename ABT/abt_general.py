import os
import numpy as np
import cv2
import random

def main():
    img_size = 512
    radius = 10
    ## make base image
    base_img = make_color_img(img_size,0) # 0 = blue
    ### create base mask
    base_msk = create_circular_mask(img_size,img_size, radius=radius)
    base_img[base_msk != 0] = 0
    ## make target images
    target1_img = make_color_img(img_size,1) # 1 = green
    target2_img = make_color_img(img_size,2) # 2 = red
    ## make target masks
    target1_msk = create_circular_mask(img_size,img_size, radius=radius)
    target2_msk = create_circular_mask(img_size,img_size, radius=radius)
    ## create target list
    target_img_list = [target1_img, target2_img]
    target_msk_list = [target1_msk, target2_msk]
    ## apply abt
    # for i in range(200):
    for target_img, target_msk in zip(target_img_list, target_msk_list):
        abt_img, abt_msk = abt(base_img, base_msk, target_img, target_msk)
        base_img = abt_img
        base_msk = abt_msk
    ## save abt image and mask
    save_as_img(['abt_img.jpg', 'abt_msk.png'], [abt_img, abt_msk])


def abt(base_img, base_mask, target_img, target_msk):
    no_attempts = 10
    marker = 0
    for count in range(no_attempts):
        # random location in base_msk for target_msk
        shift_x, shift_y = random_shift(target_msk)
        shifted_msk = shift_msk(target_msk, shift_y, shift_x)
    
        
        # check overlap
        if not check_overlap(base_mask, shifted_msk):
            abt_img, abt_msk = create_abt_image_and_mask(base_mask, base_img, 
                                                     target_img, target_msk, shifted_msk)
            return abt_img, abt_msk # TODO: shifted_msk for .npz
        
    print('overlap: placement not possible')  
    return base_img, base_mask



#####################################################################
### helper functions for ABT
#####################################################################
def random_shift(target_mask):
    # Find the positions of the mask
    mask_y, mask_x = np.where(target_mask > 0)
    # Generate random shift
    shift_y = np.random.randint(-np.min(mask_y), target_mask.shape[0] - np.max(mask_y))
    shift_x = np.random.randint(-np.min(mask_x), target_mask.shape[1] - np.max(mask_x))

    return shift_y, shift_x


def shift_msk(target_mask, shift_y, shift_x):
    y_coords, x_coords = np.where(target_mask > 0)
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


##################################################################
### helper functions for main                                  ###
##################################################################
def make_color_img(img_size,color):
    target_img = np.zeros((img_size,img_size,3), np.uint8)
    target_img[:,:,color] = 255
    return target_img

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return np.uint8(mask*255)

def save_as_img(file_names, imgs):
    for file_name, img in zip(file_names, imgs):
        cv2.imwrite(file_name, img)


if __name__ == '__main__':
    
    main()