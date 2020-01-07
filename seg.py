import glob
from tools.test import *

parser = argparse.ArgumentParser(description='CS172 Final project')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--img_path', default='None', help='img mode')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--video', default='None', help='video mode')
args = parser.parse_args()

def motion_blur(image, degree=50, angle=60):
    
    image = np.array(image, dtype=np.double)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    #blurred = cv2.GaussianBlur(image,(19,19),10,10)
    # convert to uint8
    blurred = np.array(blurred, dtype=np.double)
    
    return blurred

if __name__ == '__main__':
    cfg = load_config(args)
    if (args.video == 'None' and args.img_path == 'None'):
        exit('Please input img path or an video')
    if (args.video != 'None' and args.img_path != 'None'):
        exit('You can only chose either img mode or video mode')

    ################ Cite from demo.py of SaimMask #############
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)  
    ################ Cite from demo.py of SaimMask #############
    #video_name = args.video.split('/')[-1]
    #out_path = "output/" + video_name
    #fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    original_root = 'origin/'
    mask_root = 'mask/'
    if (args.video != 'None'):
        video = cv2.VideoCapture(args.video)
        if (video.grab()):
            flag, frame = video.retrieve()
            try:
                #size = frame.shape
                init_rect = cv2.selectROI('SiamMask', frame, False, False)
                x, y, w, h = init_rect
            except:
                exit()
            #out_video = cv2.VideoWriter(out_path, fourcc, 12, size[0:2])

        num = 0
        while(True):
            if num == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker

            elif num > 0:  # tracking
                flag, frame = video.read()
                if (not flag):
                    break
                ################ Cite from demo.py of SaimMask #############
                state = siamese_track(state, frame, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                ################ Cite from demo.py of SaimMask #############
                mask = mask + 0

                original_path = original_root + str(num) + ".jpg"
                cv2.imwrite (original_path, frame)
                
                cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                mask_path = mask_root + str(num) + ".jpg"
                cv2.imwrite (mask_path, mask)


                #out_video.write(frame)

            num += 1
            
    elif (args.img_path != 'None'):
        img_files = sorted(glob.glob(join(args.img_path, '*.jp*')))
        ims = [cv2.imread(imf) for imf in img_files]
        try:
            init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
            x, y, w, h = init_rect
        except:
            exit()

        toc = 0
        for f, im in enumerate(ims):
            tic = cv2.getTickCount()
            if f == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            elif f > 0:  # tracking
                state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                im[:, :, 0] = (mask > 0) * 0 + (mask == 0) * im[:, :, 0]
                im[:, :, 1] = (mask > 0) * 0 + (mask == 0) * im[:, :, 1]
                im[:, :, 2] = (mask > 0) * 0 + (mask == 0) * im[:, :, 2]
                cv2.imshow('SiamMask', im)
                key = cv2.waitKey(1)
                if key > 0:
                    break