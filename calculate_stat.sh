#!/bin/bash
#SBATCH --job-name calc_stat
#SBATCH -A g34
#SBATCH -t 8:00:00
#SBATCH -c 64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p long

# Calculate vertical-level-wise min and max for each variable

DATASET_PATH="/scratch/lhuang/IAC/reference.0.25"
STAT_PATH="/scratch/lhuang/IAC/reference.0.25/stats"

mkdir -p $STAT_PATH
mkdir -p $STAT_PATH/tmp
rm $STAT_PATH/P201609_*.nc
ls $DATASET_PATH/P201609* | parallel --progress -j 16 cdo fldmin {} $STAT_PATH/tmp/{/}_min
cdo -P 16 ensmin $STAT_PATH/tmp/*_min $STAT_PATH/P201609_min.nc
rm $STAT_PATH/tmp/*_min
ls $DATASET_PATH/P201609* | parallel --progress -j 16 cdo fldmax {} $STAT_PATH/tmp/{/}_max
cdo -P 16 ensmax $STAT_PATH/tmp/*_max $STAT_PATH/P201609_max.nc
rm $STAT_PATH/tmp/*_max
rm -r $STAT_PATH/tmp
#cdo -P 4 fldmin -ensmin $DATASET_PATH/P201609* $STAT_PATH/P201609_min.nc
#cdo -P 4 fldmax -ensmax $DATASET_PATH/P201609* $STAT_PATH/P201609_max.nc

#DATASET_PATH="/scratch/lhuang/IAC/3D_alllevs_025/2016"
#STAT_PATH="/scratch/lhuang/IAC/3D_alllevs_025/2016/stats"
#mkdir -p $STAT_PATH

#for MONTH in "08" "09" "10"
#do
#    echo "Processing month $MONTH"
#    cdo -P 32 fldmin -timmin $DATASET_PATH/$MONTH/Z_1h_HRplev_alllevs_025_2016_$MONTH.nc $STAT_PATH/Z_1h_HRplev_alllevs_025_2016_${MONTH}_min.nc
#    cdo -P 32 fldmax -timmax $DATASET_PATH/$MONTH/Z_1h_HRplev_alllevs_025_2016_$MONTH.nc $STAT_PATH/Z_1h_HRplev_alllevs_025_2016_${MONTH}_max.nc
#done
