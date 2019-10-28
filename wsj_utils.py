import os
import errno
import yaml
import numpy as np
import utils
import config
import soundfile as sf

def compute_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2)**2))

def create_distance_mat(array_geometry):
    mic_count = len(array_geometry)
    distance_mat = np.zeros((mic_count, mic_count))
    for mic1 in np.arange(0, mic_count):
        for mic2 in np.arange(0, mic1+1):
            if mic1 != mic2:
                distance_mat[mic1, mic2] = np.abs(array_geometry[mic1]- \
                        array_geometry[mic2])
                distance_mat[mic2, mic1] = distance_mat[mic1, mic2]
    return distance_mat

class SourceManager(object):
    def __init__(self, _id, # speech id
                 speech_file, # speech file
                 rir_obj,
                 rir_file, # RIR between the source and mics
                 mic_delay): # distance between source and mic mid point
        self._id = _id
        self.speech_file = speech_file
        if os.path.exists(rir_file):
            final_rir_file = rir_file
        elif os.path.exists(rir_file+'.gz'):
            final_rir_file = rir_file+'.gz'
        else:
            raise FileNotFoundError(errno.ENOENT, \
                    os.strerror(errno.ENOENT), rir_file)
        self.rir_file = final_rir_file
        self.rir_obj = rir_obj
        self.speech = None
        self.rir_data = None
        self.mic_delay = mic_delay

    def read_data(self):
        '''
        Read the speech and rir files if not read
        '''
        if self.speech is None:
            self.speech, _ = sf.read(self.speech_file)
        if self.rir_data is None:
            self.rir_data = np.loadtxt(self.rir_file)

    def reverberate(self):
        self.read_data()
        return utils.make_multi_channel_reverb(self.speech, self.rir_data).T

    def reverberate_with_early_echoes(self):
        self.read_data()
        speech = self.speech
        rir_data = self.rir_data
        mic_reverb = []
        for mic_idx, mic_delay in enumerate(self.mic_delay):
            mic_sample_len = int(round(mic_delay*config.SAMPLING_RATE) + \
                config.EARLY_ECHOES * config.SAMPLING_RATE)
            mic_rir = rir_data[:mic_sample_len, mic_idx]
            mic_reverb.append(utils.make_single_channel_reverb(speech, mic_rir))
        return np.array(mic_reverb).T

    def reverberate_with_direct_sound(self):
        self.read_data()
        speech = self.speech
        rir_data = self.rir_data
        mic_reverb = []
        for mic_idx, mic_delay in enumerate(self.mic_delay):
            mic_sample_len = int(round(mic_delay*config.SAMPLING_RATE)) + 3
            mic_rir = rir_data[:mic_sample_len, mic_idx]
            mic_reverb.append(utils.make_single_channel_reverb(speech, mic_rir))
        return np.array(mic_reverb).T


class WsjFileManager(object):

    def __init__(self, speech_id_list, wsj_mix_base_path,\
            source_count, noise_list, rir_list, \
            speech_key_start, speech_key_end):
        self.wsj_mix_base_path = wsj_mix_base_path
        self.noise_list = utils.parse_list(noise_list)
        self.speech_keys = utils.parse_list(speech_id_list)
        self.trim_speech_keys(speech_key_start, speech_key_end)
        self.speech_file_count = len(self.speech_keys)
        self.yaml_fid = open(rir_list, 'r')
        self.rir_file = rir_list
        self.yaml_iter = yaml.safe_load_all(self.yaml_fid)
        self.current_idx = 0
        self.source_count = source_count
        self.source_ids = np.arange(source_count)
        self.id = 0
        self.noise_file_cnt = len(self.noise_list)
        self.speech_with_enough_segments = 4

    def trim_speech_keys(self, start, end):
        self.speech_keys = self.speech_keys[start:end]

    def update_idx(self):
        self.current_idx += 1

    def __read_yaml_data(self):
        return next(self.yaml_iter, None)

    def get_src_int(self):
        raise NotImplementedError

    def create_src_inter_obj(self, rir):
        raise NotImplementedError

    def get_rir(self):
        rir = self.__read_yaml_data()
        return rir

    def pick_random_noise_file(self):
        noise_file_idx = np.random.randint(self.noise_file_cnt)
        return self.noise_list[noise_file_idx]


    def read_audio_file(self, file_name, start=None, end=None):
        if start is None:
            data, _ = sf.read(file_name, always_2d=True)
        else:
            data, _ = sf.read(file_name, always_2d=True, start=start, stop=end)
        return data

    def get_noise_data(self, seg_size, reject_file=None):
        full_noise = None
        noise_file = self.pick_random_noise_file()
        noise = self.read_audio_file(noise_file)
        noise_len = noise.shape[0]
        if full_noise is None:
            full_noise = np.zeros((seg_size, noise.shape[1]))
            remaining = seg_size
            start_at = 0
        while True:
            noise_len = noise.shape[0]
            if noise_len >= remaining:
                # Chime 5 channels notations are opposite to mine
                full_noise[start_at:start_at+remaining,0] = noise[:remaining,3]
                full_noise[start_at:start_at+remaining,1] = noise[:remaining,2]
                full_noise[start_at:start_at+remaining,2] = noise[:remaining,1]
                full_noise[start_at:start_at+remaining,3] = noise[:remaining,0]
                return full_noise.T
            full_noise[start_at:start_at+noise_len,:] = noise
            start_at += noise_len
            remaining -= noise_len

    def create_sources(self):
        rir = self.get_rir()
        return self.create_source_obj(rir)

    def get_delay(self, rir, src_ids):
        mics = rir['mic']['pos']
        #src = rir['mic/src']
        delay =  np.zeros((len(mics), len(src_ids)))
        for _mic_ in mics:
            for src_ in src_ids:
                mic_pos = np.array(rir['mic']['pos'][_mic_])
                src_pos = np.array(rir['src'][src_]['pos'])
                delay[_mic_][src_] = compute_dist(mic_pos, src_pos)/config.SOUND_SPEED
        return delay

    def get_source_filepath(self, wav_id, src_number):
        src_path = self.wsj_mix_base_path + '/s' + str(src_number) \
                + '/' + wav_id
        assert os.path.exists(src_path), 'File not found'
        return src_path

    def get_source_filepaths(self, wav_id):
        all_src_paths = []
        for src_id in range(self.source_count):
            src_id += 1
            all_src_paths.append(self.get_source_filepath(wav_id, src_id))
        return all_src_paths


    def create_source_obj(self, rir):
        src_numbers = self.source_ids
        if self.current_idx >= len(self.speech_keys):
            return None
        src_id = self.speech_keys[self.current_idx]
        self.update_idx()
        sources = []
        delay = self.get_delay(rir, src_numbers).T
        for src_number, _delay_ in zip(src_numbers, delay):
            src_path = self.get_source_filepath(src_id, src_number+1)
            src_rir_path = rir['rir_base_path'] + '/' + rir['src'][src_number]['rir_name']
            src = SourceManager(src_id, src_path, rir, src_rir_path, _delay_)
            sources.append(src)
        return sources

    def __del__(self):
        self.yaml_fid.close()
