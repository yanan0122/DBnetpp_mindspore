import numpy as np

from mindspore import Tensor, context, nn, ops
import mindspore as ms

# Input:
#             pred: A dict which contains predictions.
#                 thresh: The threshold prediction
#                 binary: The text segmentation prediction.
#                 thresh_binary: Value produced by `step_function(binary - thresh)`.
#             batch:
#                 gt: Text regions bitmap gt.
#                 mask: Ignore mask,
#                     pexels where value is 1 indicates no contribution to loss.
#                 thresh_mask: Mask indicates regions cared by thresh supervision.
#                 thresh_map: Threshold gt.

class L1BalanceCELoss(nn.Cell):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred, gt, gt_mask, thresh_map, thresh_mask):

        bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask, False)

        if 'thresh' in pred:

            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)

            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)

            loss = dice_loss + self.l1_scale * l1_loss + bce_loss_output * self.bce_scale

        else:

            loss = bce_loss_output

        return loss

class DiceLoss(nn.Cell):

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def construct(self, pred: Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''

        if weights is not None:
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        return loss

class MaskL1Loss(nn.Cell):

    def __init__(self):

        super(MaskL1Loss, self).__init__()

    def construct(self, pred, gt, mask):

        mask_sum = mask.sum()

        loss = ((pred[:, 0] - gt).abs() * mask).sum() / mask_sum

        return loss

class BalanceCrossEntropyLoss(nn.Cell):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):

        super(BalanceCrossEntropyLoss, self).__init__()

        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bceloss = nn.BCELoss()
        self.topk = ops.TopK(sorted=True)
        self.K = 100

    def construct(self, pred, gt, mask, return_origin=False):

        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''

        pos = (gt * mask)
        neg = (mask - pos)

        positive_count = pos.sum().astype(ms.int32)
        negative_count = min(neg.sum().astype(ms.int32),
                            (positive_count * self.negative_ratio).astype(ms.int32))

        loss = self.bceloss(pred, gt)[:, 0, :, :]

        # positive_loss = loss * pos
        # negative_loss = (loss * neg).view(-1)

        # negative_loss, _ = self.topk(negative_loss, negative_count)

        # negative_loss = negative_loss[:negative_count]

        # positive_sum = positive_loss.sum()

        # balance_loss = (positive_sum + negative_loss.sum()) /\
        #     (positive_count + negative_count + self.eps)

        # return balance_loss

        return loss

if __name__ == "__main__":

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5)

    pred = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/pred.npy"),dtype=ms.float32)
    gt = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/gt.npy"),dtype=ms.float32)
    gt_mask = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/mask.npy"), dtype=ms.float32)

    print(pred.shape, gt.shape, gt_mask.shape)

    thresh_map = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/thresh_maps.npy"), dtype=ms.float32)
    thresh_mask = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/thresh_masks.npy"), dtype=ms.float32)

    # diceloss = DiceLoss()
    # print(diceloss.construct(pred,gt,gt_mask,None))

    # maskl1loss = MaskL1Loss()
    # print(maskl1loss.construct(pred,gt,mask))

    balance_loss = BalanceCrossEntropyLoss()
    print(balance_loss.construct(pred,gt,gt_mask,False))


    # pred_dict = {}
    # pred_dict['binary'] = pred
    # pred_dict['thresh'] = pred

    # l1balanceloss = L1BalanceCELoss()
    # print(l1balanceloss.construct(pred_dict, gt, gt_mask, thresh_map, thresh_mask))
