# cyclegan-cbct
Testing of a cycleGAN approach for joint MR-CBCT segmentation

## installation

1. Install conda
2. conda --create --name tf2
3. conda activate tf2
4. conda install tensorflow=2.0.0
5. pip install tensorflow_addons

## run
python cyclegan-cbct/experiment.py -c test_gan/experiments/zzh_horse2zebra_config.py -d test_gan/data/horse2zebra/ -e test_gan/experiments/
