# thesis_hgoo9099

## Datasets
The datasets must be downloaded manually before running the code. Then, the datalist generation script must be run (located inside the generate_list folder) to generate the .txt files required for the program to read the data.
A scale factor should also be included as a command line arg, which will scale the size of the dataset (only for the .txt files, it will not remove examples).

*usage: python3 dataset_list_gen.py <scale factor between 0 and 1 - e.g. 0.5>*

This will generate a list with the absolute paths to each of the examples, the folders containing these lists should be moved to the './data/' folder, but the data itself should not move.

## Main Code Pipeline

### Source domain training

To train on the source domain, run the following with the appropriate dataset name (e.g. office_1.0) from inside the './uda/' folder:
*python3 image_source.py --gpu_id 0 --seed 2021 --dset <dataset name> --max_epoch 100 --batch_size=32*

e.g.
*python3 image_source.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 100 --batch_size=32*

Output and log files from this are directed to the './uda/san/uda/' folder.

### Target domain training (unsupervised domain adaptation with hypothesis transfer learning)
  
To train on the target domain, run the following with the appropriate dataset name (e.g. office_1.0):
*python3 image_target.py --gpu_id 0 --seed 2021 --output <output dir> --dset <dataset name> --cls_par 0.3 --ssl 0.6 --batch_size 32*

e.g.
*python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset office_1.0 --cls_par 0.3 --ssl 0.6 --batch_size 32*

### Benchmark training

To run the benchmark script, run the following with the appropriate dataset name (e.g. office_1.0):
*python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset <dataset name> --max_epoch 15 --output_tar <target output dir> --cls_par 0.3 --ssl 0.6 --ps 0.0*

*python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 15 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0*

### UDA Boost training

To run the UDA boost script, run the following with the appropriate dataset name (e.g. office_1.0):
*python3 image_boost.py --gpu_id 0 --seed 2021 <dataset name> --max_epoch 15 --s 0 --output_tar <target output dir> --cls_par 0.3 --ssl 0.6 --ps 0.0*

*python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0*
