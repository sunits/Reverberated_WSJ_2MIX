#Path to chime3 dataset. Should contain train, eval and dev folders

CHIME5_WAV_BASE=$1
noise_dest=$2
dihard_sad_label_path=$3

mkdir -p $noise_dest/lists
for speaker in S02 S09 S03 S04 S05 S06 S07 S08 S12 S13 S16 S17 S18 S19 S20 S22 S23 S24 ;  do
    for unit in 1 2 3 4 5 6; do
        if   [[ $speaker == S09  &&  $unit ==  5 ]] || [[ $speaker == S05  &&  $unit ==  3 ]] || [[  $speaker == S22  &&  $unit ==  3  ]] ; then
            continue
        fi
        python noise_from_chime5/getNonSpeechSegments.py $CHIME5_WAV_BASE  ${speaker}_U0${unit}  $dihard_sad_label_path/${speaker}_U0${unit}.lab $noise_dest || exit 1;
    done
done

# create lists
#Train set
for spk_id in S02 S03 S04 S05 S06 S07 S08 S09 S12; do
    ls $noise_dest/$spk_id* >> $noise_dest/lists/tr
done
#CV set
for spk_id in  S13 S16 S17 S18 S19 S20; do
    ls $noise_dest/$spk_id* >> $noise_dest/lists/cv
done
#Test set
for spk_id in   S22 S23 S24; do
    ls $noise_dest/$spk_id* >> $noise_dest/lists/tt
done
