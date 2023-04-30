python3 image_source.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 100 --s 0 --batch_size=32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset office_1.0 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset office_0.75 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset office_0.5 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset office_0.25 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021  --dset office_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;

python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0; 
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset office_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;

python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset office_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;


python3 image_source.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 100 --s 0 --batch_size=32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_1.0 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;

python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.75 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.5 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.25 --s 0 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021  --dset OfficeHome_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;

python3 image_source.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 100 --s 1 --batch_size=32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_1.0 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.5 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --ps 0.0;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.25 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_mixmatch.py --gpu_id 0 --seed 2021  --dset OfficeHome_0.25 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;


python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.75 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.5 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.25 --s 1 --cls_par 0.3 --ssl 0.6 --batch_size 32;