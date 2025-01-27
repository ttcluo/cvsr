import cv2
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import argparse
from basicsr.archs.CVSR2_arch import CVSR2
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, imwrite, tensor2img,get_time_str
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    #parser.add_argument('config', help='test config file path')
    parser.add_argument('--folder_type',type=str,default="vimeo_test" , help='folder_name')
    parser.add_argument('--vimeo', type=str, help='index corresponds to the first frame of the sequence')

    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    # -------------------- Configurations -------------------- #
    args = parse_args()
    device = torch.device('cuda',args.device)
    save_imgs = False
    # set suitable value to make sure cuda not out of memory
    # for vimeo90K dataset, we load the whole clip at once
    interval = 7
    # which channel is used to evaluate
    test_y_channel = True
    # measure the quality of center frame
    center_frame_only = True
    # flip sequence
    flip_seq = False
    crop_border = 0
    # model
    model_path = 'experiments/PSRT_Reccurrent/PSRT_Vimeo.pth'  # noqa E501
    # test data
    test_name = f'vid4{interval}-woflip'

    lr_folder = '/userhome/BasicSR/Vid4/Noise_0d05'
    gt_folder = '/userhome/BasicSR/Vid4/GT'
    save_folder = f'results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)
    
    # logger
    log_file = osp.join(save_folder, f'psnr_ssim_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    # set up the models
    model = CVSR2(num_feat=64, num_block=30)

    #model.load_state_dict(torch.load(model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    # load video names
    #vimeo_motion_txt = '/data/ssw/vimeo90k/meta_info_Vimeo90K_test_GT_part3.txt'
    avg_psnr_l = []
    avg_ssim_l = []

    subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_folder, '*')))

    # for each subfolder
    subfolder_names = []
    for subfolder, subfolder_gt in zip(subfolder_l, subfolder_gt_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_names.append(subfolder_name)

        # read lq and gt images
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)

        # calculate the iter numbers
        length = len(imgs_lq)

        # flip seq
        avg_psnr = 0
        avg_ssim = 0
        # inference
        name_idx = 0
        imgs_lq = imgs_lq.unsqueeze(0).to(device)
        # for i in range(iters):
        #     min_id = min((i + 1) * interval, length)
        #     lq = imgs_lq[:, i * interval:min_id, :, :, :]
        phi = torch.randint(0,360,(1,))
        phi = (phi/360.0)*3.14159
        phi = phi.view(1,1,1,1)
        phi = phi.cuda()
        
        with torch.no_grad():
            outputs = model(imgs_lq,phi).squeeze(0)
        # convert to numpy image
        for idx in range(outputs.shape[0]):
            img_name = imgnames[name_idx] + '.png'
            output = tensor2img(outputs[idx][0:3,:,:], rgb2bgr=True, min_max=(0, 1))
            # read GT image
            img_gt = cv2.imread(osp.join(subfolder_gt, img_name), cv2.IMREAD_UNCHANGED)
            crt_psnr = psnr_ssim.calculate_psnr(
                output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            crt_ssim = psnr_ssim.calculate_ssim(
            output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            # save
            if save_imgs:
                imwrite(output, osp.join(save_folder, subfolder_name, f'{img_name}'))
            avg_psnr += crt_psnr
            avg_ssim += crt_ssim
            logger.info(f'{subfolder_name}--{img_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')
            name_idx += 1

        avg_psnr /= name_idx
        logger.info(f'name_idx:{name_idx}')
        avg_ssim /= name_idx
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)

    for folder_idx, subfolder_name in enumerate(subfolder_names):
        logger.info(f'Folder {subfolder_name} - Average PSNR: {avg_psnr_l[folder_idx]:.6f} dB. Average SSIM: {avg_ssim_l[folder_idx]:.6f}.')

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f}  '
    f'for {len(subfolder_names)} clips. ')

if __name__ == '__main__':

    main()
