nohup ./run.sh > run.log 2>&1 &
919258
nohup ./run25_mesorch.sh > run_25_mesorch.log 2>&1 &
nohup ./run800_mesorch.sh > run_800_mesorch.log 2>&1 &
nohup ./run25_ffdn.sh > run_25_ffdn.log 2>&1 &
nohup ./run800_ffdn.sh > run_800_ffdn.log 2>&1 &
nohup ./run2_2.sh > run_2_2.log 2>&1 &
2004395
nohup ./run3.sh > run_3.log 2>&1 &

nohup /home/59435a/Forensic_hub/test_create_dataset_25.sh > run_25.log 2>&1 &
nohup /home/59435a/Forensic_hub/test_create_dataset_800.sh > run_800.log 2>&1 &
nohup /home/59435a/Forensic_hub/test_create_dataset_test.sh > run_test.log 2>&1 &


# Test
nohup ./run_test.sh > run_test.log 2>&1 &
nohup ./run_test_all.sh > run_test_all.log 2>&1 &
nohup ./run_ffdn_test_not_compress.sh > run_ffdn_test_not_compress.log 2>&1 &
nohup ./run_ffdn_test_recompress_25.sh > run_ffdn_test_recompress_25.log 2>&1 &
nohup ./run_ffdn_test_recompress_800.sh > run_ffdn_test_recompress_800.log 2>&1 &
nohup ./run_mesorch_test_not_compress.sh > run_mesorch_test_not_compress.log 2>&1 &
nohup ./run_mesorch_test_recompress_25.sh > run_mesorch_test_recompress_25.log 2>&1 &
nohup ./run_mesorch_test_recompress_800.sh > run_mesorch_test_recompress_800.log 2>&1 &