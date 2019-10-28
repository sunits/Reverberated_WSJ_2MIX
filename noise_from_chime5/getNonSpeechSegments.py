import sys
import os
import soundfile as sf
import numpy as np
#import ipdb
import argparse

MIN_NOISE_TIME_THRESHOLD = 3
FRAME_DROP_PERCENTAGE_THRESHOLD = 5


def getNonSpeechSegments(full_file_path, start_time, end_time):
    wav_data = []
    for idx in range(4):
        ch_file_name = full_file_path + '.CH' +str(idx+1) + '.wav'
        start_samples = int(start_time * 16000)
        end_samples = int(end_time * 16000)
        wav_data.append(sf.read(ch_file_name, start=start_samples, stop=end_samples)[0])
    wav_data = np.array(wav_data).T
    return wav_data

def removeFrameDrops(data):
    sample_len = data.shape[0]
    drop_len = np.sum(data[:,0] == 0)
    drop_percentage = 100 * drop_len / sample_len
    if drop_percentage > FRAME_DROP_PERCENTAGE_THRESHOLD:
        return True
    return False

def save(file_name, data, dest):
    file_name = dest + '/' + file_name
    sf.write(file_name, data, 16000)

def getFilePath(wav_base, file_name):
    train_path = wav_base + '/train/' + file_name + '.CH1.wav'
    dev_path = wav_base + '/dev/' + file_name + '.CH1.wav'
    if os.path.exists(train_path):
        path = train_path
    elif os.path.exists(dev_path):
        path = dev_path
    else:
        print('*'*50)
        print(file_name + ' File does not exists')
        print('*'*50)
        exit(1)
    return os.path.dirname(path)

def main():
    parser = argparse.ArgumentParser('Get noise segments from CHIME 5')
    parser.add_argument('wav_base', help='Base path to CHIME5 dataset')
    parser.add_argument('wav_id', help='session id with device id S02_U01')
    parser.add_argument('wav_sad', help='Path to sad file')
    parser.add_argument('dest', help='Path to save the extracted files')
    args = parser.parse_args()

    wav_id = args.wav_id
    wav_sad = args.wav_sad
    print(wav_id, wav_sad)

    filepath = getFilePath(args.wav_base, wav_id)
    filepath += '/' + wav_id
    #ipdb.set_trace()
    sampling_rate = sf.info(filepath+'.CH1.wav').samplerate
    assert sampling_rate == 16000, 'Sampling rate is not 16k'
    with open(wav_sad) as fid:
        noise_start = float(next(fid).strip().split()[1])
        for ele in fid:
            speech_start, speech_end = [float(tmp) for tmp in ele.strip().split()[:2]]
            if speech_start - noise_start <  MIN_NOISE_TIME_THRESHOLD:
                noise_start = speech_end
                continue
            noise = getNonSpeechSegments(filepath, noise_start, speech_start)
            if removeFrameDrops(noise):
                noise_start = speech_end
                continue
            noise_file_name = wav_id + '_' + str(noise_start) + '_' + str(speech_start) + '.wav'
            save(noise_file_name, noise, args.dest)
            noise_start = speech_end

main()
