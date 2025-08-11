import sys

sys.path.append('./')
from .pylung.annotation import *
import os
import numpy as np
from tqdm import tqdm
import sys
import nrrd
import SimpleITK as sitk
import cv2
from config import config
from glob import glob


def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def xml2mask(xml_file):
    header, annos = parse(xml_file)
    ctr_arrs = []
    for i, reader in enumerate(annos):
        for j, nodule in enumerate(reader.nodules):
            ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    ctr_arr.append([z, roi_xy[1], roi_xy[0]])
            ctr_arrs.append(ctr_arr)
    seriesuid = header.series_instance_uid
    return seriesuid, ctr_arrs


def annotation2masks(annos_dir, save_dir):
    files = find_all_files(annos_dir, '.xml')
    for f in tqdm(files, total=len(files)):
        try:
            seriesuid, masks = xml2mask(f)
            np.save(os.path.join(save_dir, '%s' % (seriesuid)), masks)
        except:
            print("Unexpected error:", sys.exc_info()[0])


def arr2mask(arr, reso):
    mask = np.zeros(reso)
    arr = arr.astype(np.int32)
    mask[arr[:, 0], arr[:, 1], arr[:, 2]] = 1

    return mask


def arrs2mask(img_dir, ctr_arr_dir, save_dir):
    mhd_list = []
    for i in range(10):
        mhd_list += glob(os.path.join(img_dir + 'subset' + str(i) + '/*.mhd'))
    # pids = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.mhd')]
    pids = [f[:-4] for f in mhd_list]
    cnt = 0
    consensus = {1: 0, 2: 0, 3: 0, 4: 0}

    for k in consensus.keys():
        if not os.path.exists(os.path.join(save_dir, str(k))):
            os.makedirs(os.path.join(save_dir, str(k)))
    # 用于存储没有肺结节的CT扫描
    # if not os.path.exists(os.path.join(save_dir, str(0))):
    #     os.makedirs(os.path.join(save_dir, str(0)))

    for pid in tqdm(pids, total=len(pids)):

        img, origin, spacing = load_itk_image(os.path.join(img_dir, '%s.mhd' % (pid)))
        pid = os.path.split(pid)[-1]
        # ctr_arrs为三维list
        ctr_arrs = np.load(os.path.join(ctr_arr_dir, '%s.npy' % (pid)), allow_pickle=True)
        cnt += len(ctr_arrs)

        nodule_masks = []  # 肺结节组成的list,其中每个元素代表一个与img同等大小的mask
        for ctr_arr in ctr_arrs:
            # 按照顺序取出四个医生标注的所有肺结节：ctr_arr为二维list（N*3）
            z_origin = origin[0]
            z_spacing = spacing[0]
            ctr_arr = np.array(ctr_arr)
            ctr_arr[:, 0] = np.absolute(ctr_arr[:, 0] - z_origin) / z_spacing
            ctr_arr = ctr_arr.astype(np.int32)

            mask = np.zeros(img.shape, dtype=np.uint8)

            for z in np.unique(ctr_arr[:, 0]):
                ctr = ctr_arr[ctr_arr[:, 0] == z][:, [2, 1]]
                ctr = np.array([ctr], dtype=np.int32)
                mask[z] = cv2.fillPoly(mask[z], ctr, color=(1,) * 1)
            nodule_masks.append(mask)

        i = 0
        visited = []  # 存储已访问的肺结节编号
        d = {}
        masks = []  # 存储合并重复标注后得到的肺结节集合
        while i < len(nodule_masks):
            # If mached before, then no need to create new mask
            if i in visited:
                i += 1
                continue
            same_nodules = []  # 存储医生之间达成共识的重叠肺结节
            mask1 = nodule_masks[i]
            same_nodules.append(mask1)
            d[i] = {}
            d[i]['count'] = 1
            d[i]['iou'] = []

            # Find annotations pointing to the same nodule
            for j in range(i + 1, len(nodule_masks)):
                # if not overlapped with previous added nodules
                if j in visited:
                    continue
                mask2 = nodule_masks[j]
                iou = float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum()

                if iou > 0.4:
                    visited.append(j)
                    same_nodules.append(mask2)
                    d[i]['count'] += 1
                    d[i]['iou'].append(iou)

            masks.append(same_nodules)
            i += 1

        for k, v in d.items():
            if v['count'] > 4:
                print('WARNING:  %s: %dth nodule, iou: %s' % (pid, k, str(v['iou'])))
                v['count'] = 4
            consensus[v['count']] += 1

        # number of consensus
        num = np.array([len(m) for m in masks])
        num[num > 4] = 4
        # 原始代码中跳过了没有肺结节的CT切片
        # TODO:需将不含肺结节的CT图像也添加到数据集中以进行公平的性能比较
        # if len(num) == 0:
        #     mask = np.zeros(img.shape, dtype=np.uint8)
        #     nrrd.write(os.path.join(save_dir, str(0), pid), mask)
        #     continue
        # Iterate from the nodules with most consensus
        # for n in range(num.max(), 0, -1):
        for n in range(4, 0, -1):
            mask = np.zeros(img.shape, dtype=np.uint8)

            for i, index in enumerate(np.where(num >= n)[0]):
                same_nodules = masks[index]
                m = np.logical_or.reduce(same_nodules)
                mask[m] = i + 1  # 为每个肺结节分配一个编号，后续通过最大编号确定CT中的肺结节数量和位置
            nrrd.write(os.path.join(save_dir, str(n), pid), mask)

    #         for i, same_nodules in enumerate(masks):
    #             cons = len(same_nodules)
    #             if cons > 4:
    #                 cons = 4
    #             m = np.logical_or.reduce(same_nodules)
    #             mask[m] = i + 1
    #             nrrd.write(os.path.join(save_dir, str(cons), pid), mask)

    print(consensus)
    print(cnt)


if __name__ == '__main__':
    annos_dir = config['annos_dir']
    img_dir = config['data_dir']
    ctr_arr_save_dir = config['ctr_arr_save_dir']
    mask_save_dir = config['mask_save_dir']

    os.makedirs(ctr_arr_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # annotation2masks(annos_dir, ctr_arr_save_dir)
    arrs2mask(img_dir, ctr_arr_save_dir, mask_save_dir)
