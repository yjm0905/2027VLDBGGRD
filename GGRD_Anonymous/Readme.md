<div align="center">
   
# Graph-Guided Reconstruction Diffusion Model (GGRD)

</div>



## 	Get Started

1. Install the environment:
```bash
conda create -n GGRD python=3.8.20
conda activate GGRD
pip install -r requirements.txt
```
2. You must manually create a "dataset" folder yourself (in the same directory as main.py) and download data and place the dataset in such a directory.For the SWaT dataset, you must apply through official channels.，you can apply for it by following its [official tutorial](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
```tex
dataset
——SMD
————SMD_test_data.npy
————SMD_test_date.npy
————SMD_test_label.npy
————SMD_train_data.npy
————SMD_train_date.npy
——PSM
————PSM_test_data.npy
————PSM_test_date.npy
————PSM_test_label.npy
————PSM_train_data.npy
————PSM_train_date.npy
——SWaT
————SWaT_test_data.npy
————SWaT_test_date.npy
————SWaT_test_label.npy
————SWaT_train_data.npy
————SWaT_train_date.npy
```
  

3. Train and evaluate. You can reproduce the experiment results as follows:

   ```shell
   bash ./script/run.sh
   ```



