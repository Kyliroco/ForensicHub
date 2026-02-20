rm -rf data/dataset_pub data/dataset_pub_test
echo fin de suppression
python create_dataset.py --config datasets_800.txt --out-train data/dataset_pub --seed 42 --out-test data/dataset_pub_test