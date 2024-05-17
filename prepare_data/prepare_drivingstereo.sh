# download proxy labels from gdrive
gdown --fuzzy https://drive.google.com/file/d/13OSlqnHWBxIOk_aJya3sOfAwhxQTKE0p/view?usp=sharing -O sequences/drivingstereo/
unzip sequences/drivingstereo/*zip -d sequences/drivingstereo/ | tqdm --desc extracted --unit files --unit_scale 
rm sequences/drivingstereo/*zip 

for ENTRY in `cat prepare_data/drivingstereo_sequences_to_download.txt`; do

    SEQ=`echo $ENTRY | cut -d',' -f1`
    LEFT=`echo $ENTRY | cut -d',' -f2`
    RIGHT=`echo $ENTRY | cut -d',' -f3`
    GT=`echo $ENTRY | cut -d',' -f4`

    # download left images from gdrive
    gdown --fuzzy $LEFT -O sequences/drivingstereo/train-left-image/
    unzip sequences/drivingstereo/train-left-image/*zip -d sequences/drivingstereo/train-left-image/ | tqdm --desc extracted --unit files --unit_scale 
    rm sequences/drivingstereo/train-left-image/*zip

    # download right images from gdrive
    gdown --fuzzy $RIGHT -O sequences/drivingstereo/train-right-image/
    unzip sequences/drivingstereo/train-right-image/*zip -d sequences/drivingstereo/train-right-image/ | tqdm --desc extracted --unit files --unit_scale 
    rm sequences/drivingstereo/train-right-image/*zip

    # download gt disparities from gdrive
    gdown --fuzzy $GT -O sequences/drivingstereo/train-disparity-map/
    unzip sequences/drivingstereo/train-disparity-map/*zip -d sequences/drivingstereo/train-disparity-map/ | tqdm --desc extracted --unit files --unit_scale 
    rm sequences/drivingstereo/train-disparity-map/*zip

    mkdir -p sequences/drivingstereo_temp
    for i in `ls sequences/drivingstereo/train-left-image/$SEQ*jpg`; do 
        name=`echo $i | rev | cut -d'/' -f1 | rev | cut -d'.' -f1`;
        mv $i sequences/drivingstereo_temp/$name.image_02.jpg;
    done

    for i in `ls sequences/drivingstereo/train-right-image/$SEQ*jpg`; do 
        name=`echo $i | rev | cut -d'/' -f1 | rev | cut -d'.' -f1`;
        mv $i sequences/drivingstereo_temp/$name.image_03.jpg;
    done

    for i in `ls sequences/drivingstereo/train-disparity-map/$SEQ*png`; do 
        name=`echo $i | rev | cut -d'/' -f1 | rev | cut -d'.' -f1`;
        mv $i sequences/drivingstereo_temp/$name.groundtruth.png;
    done

    for i in `ls sequences/drivingstereo/train-sgm*/$SEQ*png`; do 
        name=`echo $i | rev | cut -d'/' -f1 | rev | cut -d'.' -f1`;
        mv $i sequences/drivingstereo_temp/$name.proxy.png;
    done

    mv sequences/drivingstereo_temp/* ./
    tar --sort=name -cf sequences/drivingstereo/$SEQ.tar *jpg *png
    rm -r sequences/drivingstereo_temp/
    rm *jpg
    rm *png
    rm -r sequences/drivingstereo/train*/*
done

rm -r sequences/drivingstereo/train*
