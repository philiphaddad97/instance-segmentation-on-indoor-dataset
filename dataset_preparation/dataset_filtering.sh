
# annotations filtering
python dataset_filter.py --input_json annotations/instances_train2017.json --output_json annotations/instances_train2017.json --categories chair couch bed 'dining table' window toilet tv oven sink refrigerator
python dataset_filter.py --input_json annotations/instances_val2017.json --output_json annotations/instances_val2017.json --categories chair couch bed 'dining table' window toilet tv oven sink refrigerator


# images filtering
python dataset_preparation/images_filter.py