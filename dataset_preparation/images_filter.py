import os
from pycocotools.coco import COCO


if __name__ == "__main__":

    imgIds = set([])

    coco = COCO('/content/annotations/instances_train2017.json')
    category_Ids = [ 62, 63, 65, 67, 68, 70, 72, 79, 81, 82] # Required objects IDs
    for Id in category_Ids:
      imgIds.update(coco.getImgIds(catIds=Id))

    for filename in os.listdir('/content/train2017/'):
      if filename.split('.')[0] not in imgIds:
          os.remove('dataset/train2017/' + filename)