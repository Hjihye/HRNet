import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
# import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print( 'Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
dataDir='/media/hjh/2T/app/human-pose-estimation.pytorch/data/coco'
dataType='val2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)


#initialize COCO detections api
resFile= '/media/hjh/2T/app/HRNet_jh/all_coco_val_pred.json'
anns = json.load(open(resFile))
cocoDt= cocoGt.loadRes(anns['annotations'])

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()