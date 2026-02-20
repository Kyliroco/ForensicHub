rm -rf data/dataset_pub_25 data/dataset_pub_test_25
echo fin de suppression
python create_dataset.py --config datasets_25.txt --out-train data/dataset_pub_25 --seed 42 --out-test data/dataset_pub_test_25