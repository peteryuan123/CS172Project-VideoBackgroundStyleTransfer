import cv2
import glob
import os
import numpy as np


def motion_blur(image, degree=60, angle=60):
    
    image = np.array(image, dtype=np.double)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    #blurred = cv2.GaussianBlur(image,(19,19),10,10)
    blurred = np.array(blurred, dtype=np.double)
    
    return blurred

def apply (img, mask):
    return img * mask

def main():
    transfer_path = 'data/transfered'
    origin_path = 'data/origin'
    mask_path = 'data/mask'

    file_list = os.listdir(origin_path)
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
    
    video_name = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(video_name, fourcc, 20, (640,480))

    for name in file_list:
        people_mask = cv2.imread (mask_path + '/' + name)

        back_mask = np.logical_not (people_mask)
        back_mask = back_mask + 0

        origin = cv2.imread (origin_path + '/' + name)
        transfered = cv2.imread (transfer_path + '/' + name)

        back_blurred_mask = motion_blur (back_mask)
        people_blurred_mask = motion_blur (people_mask)
        
        people = origin * people_blurred_mask
        back = transfered * back_blurred_mask
        result = people + back
        
        result_path = "data/result/" + name
        result = result.astype (np.uint8)
        cv2.imwrite (result_path, transfered)

        out_video.write (transfered)

    out_video.release()

if __name__ == '__main__':
    main()
        
