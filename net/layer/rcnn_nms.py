from net.layer.util import box_transform, box_transform_inv, clip_boxes
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap


def rcnn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)


def rcnn_decode(window, delta, weight):
    return  box_transform_inv(window, delta, weight)


def rcnn_nms(cfg, mode, inputs, proposals, logits, deltas):

    if mode in ['train',]:
        nms_pre_score_threshold = cfg['rcnn_train_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rcnn_train_nms_overlap_threshold']

    elif mode in ['valid', 'test','eval']:
        nms_pre_score_threshold = cfg['rcnn_test_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rcnn_test_nms_overlap_threshold']
        #
        # if mode in ['eval']:
        #     nms_pre_score_threshold = 0.05 # set low numbe r to make roc curve.

    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?'%mode)


    batch_size, _, depth, height, width = inputs.size() #original image width
    num_class = cfg['num_class']

    # probs     = np_sigmoid(logits.cpu().data.numpy())
    probs = F.softmax(logits).cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
    proposals = proposals.cpu().data.numpy()
    # masks = (F.sigmoid(mask_logits).cpu().data.numpy() > 0.5).astype(np.uint8)

    #non-max suppression
    detections = []
    # segments = []
    keeps = []
    for b in range(batch_size):
        detection = [np.empty((0, 9), np.float32),]

        index = np.where(proposals[:,0] == b)[0]
        if len(index)>0:
            prob  = probs[index]
            delta = deltas[index]
            proposal = proposals[index]
            # mask = masks[index]
            # cats = np.argmax(prob, 1)

            for j in range(1, num_class): #skip background
                idx = np.where(prob[:, j] > nms_pre_score_threshold)[0]
                # idx = np.where(cats == j)[0]
                if len(idx)>0:
                    p = prob[idx, j].reshape(-1, 1)
                    d = delta[idx, j]
                    # m = mask[idx, j - 1]
                    box = rcnn_decode(proposal[idx, 2:8], d, cfg['box_reg_weight'])
                    box = clip_boxes(box, inputs.shape[2:])
                    # box = clip_boxes(box, width, height)

                    # keep = filter_boxes(box, min_size = nms_min_size)
                    # num  = len(keep)
                    # if num>0:
                        # box  = box[keep]
                        # p    = p[keep]
                    js = np.expand_dims(np.array([j] * len(p)), axis=-1)
                    output = np.concatenate((p, box, js), 1)

                    if len(output) > 0:
                        output = torch.from_numpy(output).float()
                        output, keep = torch_nms(output, nms_overlap_threshold)

                    num = len(output)

                    if num > 0:
                        det = np.zeros((num, 9),np.float32)
                        det[:, 0] = b
                        det[:, 1:] = output
                        detection.append(det)
                        # segments.append(m[keep.numpy()])
                        keeps.extend(index[idx[keep.numpy()]].tolist())

        detection = np.vstack(detection)

        detections.append(detection)

    detections = Variable(torch.from_numpy(np.vstack(detections))).cuda()
    # segments = np.vstack(segments)
    return detections, keeps

#
# def get_probability(cfg, mode, inputs, proposals, logits, deltas):
#     if mode in ['train',]:
#         nms_pre_score_threshold = cfg['rcnn_train_nms_pre_score_threshold']
#         nms_overlap_threshold   = cfg['rcnn_train_nms_overlap_threshold']
#
#     elif mode in ['valid', 'test','eval']:
#         nms_pre_score_threshold = cfg['rcnn_test_nms_pre_score_threshold']
#         nms_overlap_threshold   = cfg['rcnn_test_nms_overlap_threshold']
#     else:
#         raise ValueError('rcnn_nms(): invalid mode = %s?'%mode)
#
#     num_class = cfg['num_class']
#     probs = F.softmax(logits).cpu().data.numpy()
#     deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
#     proposals = proposals.cpu().data.numpy()
#
#     for j in range(1, num_class):  # skip background
#         idx = np.where(probs[:, j] > nms_pre_score_threshold)[0]
#         if len(idx) > 0:
#             p = probs[idx, j].reshape(-1, 1)
#             d = deltas[idx, j]
#             box = rcnn_decode(proposals [idx, 2:8], d, cfg['box_reg_weight'])
#             box = clip_boxes(box, inputs.shape[2:])
#             js = np.expand_dims(np.array([j] * len(p)), axis=-1)
#             output = np.concatenate((p, box, js), 1)
#
#     return torch.from_numpy(output).cuda().float()


# def get_probability(cfg, mode, inputs, proposals, logits, deltas):
#     if mode in ['train']:
#         nms_pre_score_threshold = cfg['rcnn_train_nms_pre_score_threshold']
#         nms_overlap_threshold = cfg['rcnn_train_nms_overlap_threshold']
#     elif mode in ['valid', 'test', 'eval']:
#         nms_pre_score_threshold = cfg['rcnn_test_nms_pre_score_threshold']
#         nms_overlap_threshold = cfg['rcnn_test_nms_overlap_threshold']
#     else:
#         raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)
#
#     num_class = cfg['num_class']
#     probs = F.softmax(logits, dim=1).cpu().data.numpy()
#     deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
#     proposals = proposals.cpu().data.numpy()
#
#     print(f"Probs shape: {probs.shape}")
#     print(f"Deltas shape: {deltas.shape}")
#     print(f"Proposals shape: {proposals.shape}")
#
#     # If proposals are empty, return empty tensor
#     if proposals.size == 0:
#         print("Proposals are empty.")
#         return torch.tensor([]).cuda().float()
#
#     output_list = []
#
#     for j in range(1, num_class):  # skip background
#         idx = np.where(probs[:, j] > nms_pre_score_threshold)[0]
#
#         print(f"Class {j}: Number of valid indices: {len(idx)}")
#         print(f"Indices: {idx}")
#
#         if len(idx) > 0:
#             p = probs[idx, j].reshape(-1, 1)
#             d = deltas[idx, j]
#
#             try:
#                 boxes = proposals[idx, 2:8]  # Accessing the boxes from proposals
#             except IndexError as e:
#                 print(f"IndexError when accessing proposals: {e}")
#                 print(f"Proposals: {proposals}")
#                 return torch.tensor([]).cuda().float()
#
#             box = rcnn_decode(boxes, d, cfg['box_reg_weight'])  # Assuming rcnn_decode operates on each box individually
#             box = clip_boxes(box, inputs.shape[2:])
#             js = np.expand_dims(np.array([j] * len(p)), axis=-1)
#             class_output = np.concatenate((p, box, js), axis=1)
#             output_list.append(class_output)
#
#     if len(output_list) > 0:
#         output = np.concatenate(output_list, axis=0)
#         return torch.from_numpy(output).cuda().float()
#     else:
#         # Return a default tensor if no valid output is generated
#         return torch.tensor([]).cuda().float()
#

def get_probability(cfg, mode, inputs, proposals, logits, deltas):
    if mode in ['train']:
        nms_pre_score_threshold = cfg['rcnn_train_nms_pre_score_threshold']
        nms_overlap_threshold = cfg['rcnn_train_nms_overlap_threshold']
    elif mode in ['valid', 'test', 'eval']:
        nms_pre_score_threshold = cfg['rcnn_test_nms_pre_score_threshold']
        nms_overlap_threshold = cfg['rcnn_test_nms_overlap_threshold']
    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)

    num_class = cfg['num_class']
    probs = F.softmax(logits).cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
    proposals = proposals.cpu().data.numpy()

    print(f"Proposals shape: {proposals.shape}")
    if proposals.size == 0:
        print("No proposals found.")
        return torch.empty((0, 7)).cuda().float()  # Return an empty tensor

    output_list = []
    for j in range(1, num_class):  # skip background
        idx = np.where(probs[:, j] > nms_pre_score_threshold)[0]
        print(f"Class {j}, number of proposals passing threshold: {len(idx)}")
        if len(idx) > 0:
            p = probs[idx, j].reshape(-1, 1)
            d = deltas[idx, j]
            box = rcnn_decode(proposals[idx, 2:8], d, cfg['box_reg_weight'])
            box = clip_boxes(box, inputs.shape[2:])
            js = np.expand_dims(np.array([j] * len(p)), axis=-1)
            output = np.concatenate((p, box, js), 1)
            output_list.append(output)

    if len(output_list) == 0:
        print("No outputs generated.")
        return torch.empty((0, 7)).cuda().float()  # Return an empty tensor

    final_output = np.concatenate(output_list, axis=0)
    return torch.from_numpy(final_output).cuda().float()


#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
