import numpy as np
import cv2
import os
import argparse
import csv



#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if (len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def get_acc_v2(label_all, predict_all, classNum=2, save_path='./'):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    print("")
    print("confusion_matrix:")
    print(confusionMatrix)
    print("precision:")
    print(precision)
    print("recall:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("overall_accuracy:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    with open('{}/accuracy.txt'.format(save_path), 'w') as ff:
        ff.writelines("confusion_matrix:\n")
        ff.writelines(str(confusionMatrix)+"\n")
        ff.writelines("precision:\n")
        ff.writelines(str(precision)+"\n")
        ff.writelines("recall:\n")
        ff.writelines(str(recall)+"\n")
        ff.writelines("F1-Score:\n")
        ff.writelines(str(f1ccore)+"\n")
        ff.writelines("overall_accuracy:\n")
        ff.writelines(str(OA)+"\n")
        ff.writelines("IoU:\n")
        ff.writelines(str(IoU)+"\n")
        ff.writelines("mIoU:\n")
        ff.writelines(str(mIOU)+"\n")
        ff.writelines("FWIoU:\n")
        ff.writelines(str(FWIOU)+"\n")
    return precision, recall, f1ccore, OA, IoU, mIOU


def cal_confusion(gt_label, pre_label, n_class):
    mask = (gt_label >= 0) & (gt_label < n_class)
    confusion = np.bincount(n_class * gt_label[mask].astype(int) + pre_label[mask],
                            minlength=n_class ** 2).reshape(n_class, n_class)
    return confusion


def GetMetrics(gt_label, pre_label, n_class, save_path='./'):
    gt_label = gt_label.flatten()
    pre_label = pre_label.flatten()
    confu_mat_total = cal_confusion(gt_label, pre_label, n_class)

    # confu_mat_total = np.zeros((n_class, n_class))
    # for i in range(gt_label.shape[0]):
    #     confu_mat_total+=cal_confusion(gt_label[i].flatten(), pre_label[i].flatten(), n_class)

    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    OverallAccuracy = 0
    for i in range(class_num):
        OverallAccuracy = OverallAccuracy + confu_mat[i, i]
    OverallAccuracy = OverallAccuracy / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    Kappa = (OverallAccuracy - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    ClassF1 = np.zeros(class_num)
    ClassIoU = np.zeros(class_num)
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        ClassF1[i] = f1
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        ClassIoU[i] = iou

    MeanF1 = np.mean(ClassF1)
    MeanIoU = np.mean(ClassIoU)
    ClassRecall = np.zeros(class_num)
    ClassPrecision = np.zeros(class_num)
    if save_path is not None:
        with open('{}/accuracy.txt'.format(save_path), 'w') as f:
            f.write('OA:\t%.4f\n' % (OverallAccuracy * 100))
            f.write('kappa:\t%.4f\n' % (Kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(ClassF1) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(ClassIoU) * 100))

            print('OA:\t%.4f' % (OverallAccuracy * 100))
            print('kappa:\t%.4f' % (Kappa * 100))
            print('mf1-score:\t%.4f' % (np.mean(ClassF1) * 100))
            print('mIou:\t%.4f' % (np.mean(ClassIoU) * 100))

            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i]/raw_sum[i])*100))
                ClassPrecision[i] = float(TP[i] / raw_sum[i])
            print('precision:')
            print(ClassPrecision)
            f.write('\n')

            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i])*100))
                ClassRecall[i] = float(TP[i] / col_sum[i])
            print('recall:')
            print(ClassRecall)
            f.write('\n')

            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(ClassF1[i]) * 100))
            print('f1-score:')
            print(ClassF1)
            f.write('\n')

            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(ClassIoU[i]) * 100))
            print('Iou:')
            print(ClassIoU)
            f.write('\n')
    ClassRecall = np.asarray(ClassRecall)
    ClassPrecision = np.asarray(ClassPrecision)
    return OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision


def CM2Metric(confu_mat_total):
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    OverallAccuracy = 0
    for i in range(class_num):
        OverallAccuracy = OverallAccuracy + confu_mat[i, i]
    OverallAccuracy = OverallAccuracy / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    Kappa = (OverallAccuracy - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    ClassF1 = np.zeros(class_num)
    ClassIoU = np.zeros(class_num)
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        ClassF1[i] = f1
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        ClassIoU[i] = iou

    MeanF1 = np.mean(ClassF1)
    MeanIoU = np.mean(ClassIoU)
    ClassRecall = np.zeros(class_num)
    ClassPrecision = np.zeros(class_num)

    for i in range(class_num):
        ClassPrecision[i] = float(TP[i] / raw_sum[i])
    for i in range(class_num):
        ClassRecall[i] = float(TP[i] / col_sum[i])
    ClassRecall = np.asarray(ClassRecall)
    ClassPrecision = np.asarray(ClassPrecision)
    KEEP = 8
    ClassIoU = np.around(ClassIoU, KEEP)
    ClassF1 = np.around(ClassF1, KEEP)
    ClassRecall = np.around(ClassRecall, KEEP)
    ClassPrecision = np.around(ClassPrecision, KEEP)
    np.set_printoptions(suppress=True)
    return OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision


def dir_metric(pred_path, gt_path, pred_suffix, gt_suffix, n_class):
    files = os.listdir(pred_path)
    confu_mat_total = np.zeros((n_class, n_class))
    for item in files:
        name = item.split('.')[0]
        cur_pred = cv2.imread(os.path.join(pred_path, name+pred_suffix), 0)
        cur_gt = cv2.imread(os.path.join(gt_path, name + gt_suffix), 0)
        cur_pred = np.where(cur_pred>0, 1, 0)
        # cur_gt = np.where(cur_gt>0, 1, 0)
        cur_gt = cur_gt//255
        confu_mat_total+=cal_confusion(cur_gt.flatten(), cur_pred.flatten(), n_class)
    metrics = CM2Metric(confu_mat_total)
    names = 'OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    for i in range(len(names)):
        print(names[i]+': '+str(metrics[i]))
    # names = 'OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    paper_metrics = [metrics[0], metrics[1], metrics[2], metrics[4][1], metrics[5][1], metrics[6][1], metrics[7][1]]
    paper_metrics = [round(value * 100, 2) if isinstance(value, (int, float)) else [round(v * 100, 2) for v in value] for value in paper_metrics]

    names = 'OverallAccuracy, MeanF1, MeanIoU, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    csv_filename = os.path.join(os.path.dirname(args.pppred), 'overall_result.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(names)
        csv_writer.writerow(paper_metrics)
    print(f'Data saved to {csv_filename}')
    

if __name__ == '__main__':
    # pppred = '/home/dsj/0code_hub/cd_code/SegNeXt-main/work_dirs/segnext.base.512x512.levir.160k/test_result_big'
    # gggt = '/home/jicredt_data/dsj/CDdata/LEVIR-CD/test/label'
    # pred_suffix = '.png'
    # gt_suffix = '.png'

    parser = argparse.ArgumentParser(description='参数')
    parser.add_argument('--pppred', type=str, default='/home/dsj/0code_hub/cd_code/SegNeXt-main/work_dirs/sem_fpn_p2t_b_levir_80k_cdsa/test_result_big')
    parser.add_argument('--gggt', type=str, default='/home/jicredt_data/dsj/CDdata/LEVIR-CD/test/label')
    parser.add_argument('--pred_suffix', type=str, default='.png')
    parser.add_argument('--gt_suffix', type=str, default='.png')
    args = parser.parse_args()

    x = dir_metric(args.pppred, args.gggt, args.pred_suffix, args.gt_suffix, 2)
