def wbf(prediction,
        img,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
        ):
    """WBF on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    # Settings

    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        # if labels and len(labels[xi]):
        #     lb = labels[xi]
        #     v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
        #     v[:, :4] = lb[:, 1:5]  # box
        #     v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
        #     x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue


        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes_list, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores

        labels_list = list(x[:, 5:6].cpu())  # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        boxes_list = boxes_list.cpu().numpy()
        scores_list = list(scores.cpu())


        x1 = boxes_list[:, 0] / img.shape[3] 
        y1 = boxes_list[:, 1] / img.shape[2]
        x2 = boxes_list[:, 2] / img.shape[3]
        y2 = boxes_list[:, 3] / img.shape[2]
        boxes_list = list(np.c_[x1, y1, x2, y2])

        boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=iou_thres,
                                                      skip_box_thr=0.0, conf_type='Bayes')
        # boxes, scores, labels = weighted_boxes_fusion([list(boxes)], [list(scores)], [list(labels)], weights=None,
        #                                               iou_thr=iou_thres,
        #                                               skip_box_thr=0.0, conf_type='max')

        x1 = boxes[:, 0] * img.shape[3]  
        y1 = boxes[:, 1] * img.shape[2]
        x2 = boxes[:, 2] * img.shape[3]
        y2 = boxes[:, 3] * img.shape[2]
        boxes = list(np.c_[x1, y1, x2, y2])
        i = list(np.c_[boxes, scores, labels])
        # output = [torch.Tensor(output)]
        i = torch.Tensor(i)  
        i = i.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        output[xi] = i
    return output
