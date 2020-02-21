#PBS -l nodes=1:ppn=8:gpus=1:V100,walltime=165:00:00
#PBS -m be
source deactivate
source activate tf2.0
python cyclegan-cbct/experiment.py -c test_gan/experiments/horse2zebra_config.py -d test_gan/data/horse2zebra/ -e test_gan/experiments/
