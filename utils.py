

def cal_iou(bbx, bbx_gt):
    x1, x2, y1, y2 = bbx
    X1, X2, Y1, Y2 = bbx_gt
    s1 = (y2-y1)*(x2-x1)
    s2 = (Y2-Y1)*(X2-X1)
    jx1 = max(x1,X1)
    jx2 = min(x2,X2)
    jy1 = max(y1,Y1)
    jy2 = min(y2,Y2)
    if jx2>jx1 and jy2>jy1:
        s3 = (jx2-jx1)*(jy2-jy1)
    else:
        s3 = 0.0
    return s3/(s1+s2-s3)

def reward_func(bbx, new_bbx, bbx_gt, action):
    if action == 5:
        if cal_iou(new_bbx, bbx_gt) > 0.5:
            return 3
        else:
            return -3
    else:
        old_iou = cal_iou(bbx, bbx_gt)
        new_iou = cal_iou(new_bbx,bbx_gt)
        if(new_iou>old_iou):
            return 1
        elif(new_iou<=old_iou):
            return -1