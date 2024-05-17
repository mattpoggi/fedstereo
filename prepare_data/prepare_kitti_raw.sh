# KITTI GT depth
wget -P sequences/kitti_raw/gt/ https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip sequences/kitti_raw/gt/data_depth_annotated.zip -d sequences/kitti_raw/gt/ | tqdm --desc extracted --unit files --unit_scale 
rm sequences/kitti_raw/gt/data_depth_annotated.zip
mv sequences/kitti_raw/gt/train/2011* sequences/kitti_raw/gt/
mv sequences/kitti_raw/gt/val/2011* sequences/kitti_raw/gt/
rmdir sequences/kitti_raw/gt/train
rmdir sequences/kitti_raw/gt/val

# KITTI raw labels
gdown --fuzzy https://drive.google.com/file/d/1t9X12cYAQqJ6G8U2-XzO4oca3u2KnsYl/view?usp=sharing -O sequences/kitti_raw/sgm/
unzip sequences/kitti_raw/sgm/kitti-sgm.zip -d sequences/kitti_raw/sgm/ | tqdm --desc extracted --unit files --unit_scale 
rm sequences/kitti_raw/sgm/kitti-sgm.zip

for SEQ in `cat prepare_data/kitti_sequences_to_download.txt`; do
    echo Downloading sequence $SEQ
    wget -P sequences/kitti_raw/ https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"$SEQ"/"$SEQ"_sync.zip 
    unzip sequences/kitti_raw/"$SEQ"_sync.zip -d sequences/kitti_raw/ | tqdm --desc extracted --unit files --unit_scale  
    rm sequences/kitti_raw/"$SEQ"_sync.zip
    mkdir -p sequences/kitti_temp_folder/
    for i in `ls sequences/kitti_raw/gt/"$SEQ"_sync/proj_depth/groundtruth/image_02/*png`; do
        name=`echo $i | cut -d'.' -f1 | rev | cut -d'/' -f1 | rev`
        date=`echo $i | cut -d'/' -f4 | cut -d'_' -f1-3`
        convert -quality 92 -sampling-factor 2x2,1x1,1x1 sequences/kitti_raw/$date/"$SEQ"_sync/image_02/data/$name.png sequences/kitti_temp_folder/$name.image_02.jpg
        convert -quality 92 -sampling-factor 2x2,1x1,1x1 sequences/kitti_raw/$date/"$SEQ"_sync/image_03/data/$name.png sequences/kitti_temp_folder/$name.image_03.jpg
        cp sequences/kitti_raw/sgm/"$SEQ"_sync/image_02/data/$name.png sequences/kitti_temp_folder/$name.proxy.png
        python -W ignore prepare_data/kitti_depth2disp.py --src $i --tgt sequences/kitti_temp_folder/$name.groundtruth.png
    done
    mv sequences/kitti_temp_folder/* ./
    tar --sort=name -cf sequences/kitti_raw/"$SEQ"_sync.tar *jpg *png
    rm -r sequences/kitti_temp_folder/
    rm *jpg
    rm *png
    rm -r sequences/kitti_raw/2011*/

done

rm -r sequences/kitti_raw/sgm/
rm -r sequences/kitti_raw/gt/
