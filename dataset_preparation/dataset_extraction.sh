wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# annotations filtering
python dataset_filter.py --input_json annotations/instances_train2017.json --output_json annotations/instances_train2017.json --categories chair couch bed 'dining table' window toilet tv oven sink refrigerator
python dataset_filter.py --input_json annotations/instances_val2017.json --output_json annotations/instances_val2017.json --categories chair couch bed 'dining table' window toilet tv oven sink refrigerator
