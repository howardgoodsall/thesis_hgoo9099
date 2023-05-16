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


python3 image_source.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 100 --s 0 --batch_size 32;
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

python3 image_source.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 100 --s 2 --batch_size=32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_1.0 --s 2 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.75 --s 2 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.5 --s 2 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.25 --s 2 --cls_par 0.3 --ssl 0.6 --batch_size 32;

python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 1  --ps 0.0 --augratio 1.0 --label_smooth False;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 2  --ps 0.0 --augratio 1.0 --label_smooth False;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 3  --ps 0.0 --augratio 1.0 --label_smooth False;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 4  --ps 0.0 --augratio 1.0 --label_smooth False;


python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 2 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 2 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 2 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
 python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 0 --output_tar ckps/target/ --ssl 0.6  --ps 0.0 --augratio 2.0;
python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 25 --s 0 --output_tar ckps/target/ --ssl 0.6  --ps 0.0 --augratio 3.0 --nolog True --lr 1e-3;


python3 image_source.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 100 --s 3 --batch_size=32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_1.0 --s 3 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.75 --s 3 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.5 --s 3 --cls_par 0.3 --ssl 0.6 --batch_size 32;
python3 image_target.py --gpu_id 0 --seed 2021 --output ckps\\target\\ --dset OfficeHome_0.25 --s 3 --cls_par 0.3 --ssl 0.6 --batch_size 32;


python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_1.0 --max_epoch 15 --s 3 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.75 --max_epoch 15 --s 3 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.5 --max_epoch 15 --s 3 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;
python3 image_mixmatch.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 15 --s 3 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6  --ps 0.0;

python3 image_boost.py --gpu_id 0 --seed 2021 --dset OfficeHome_0.25 --max_epoch 30 --s 3 --output_tar ckps/target/ --ssl 0.6  --ps 0.0 --augratio 2.5  --blur True --aug_severity 3 --batch_norm True --source_classifier False --lr 1e-3 --nolog True

python3 image_source.py --gpu_id 0 --seed 2021 --dset digits --max_epoch 100 --s 0 --batch_size=32;
