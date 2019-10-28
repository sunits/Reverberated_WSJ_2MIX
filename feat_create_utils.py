import os
import yaml
import argparse
import soundfile as sf
import librosa
import numpy as np
import config
import utils
import ipdb
import h5py
import sys
import soundfile as sf
import itertools
import matplotlib.pylab as plt
extra_paths = []
extra_paths.append('/talc3/multispeech/calcul/users/ssivasankaran/experiments/code/sunit-code/global_dnn/kaldi/')
extra_paths.append('/talc3/multispeech/calcul/users/ssivasankaran/experiments/code/sunit-code/utils/')
for path in extra_paths:
    sys.path.append(path)
import readKaldiData
import ssn

def normalize_multichannel(to_normalize_data, normalize_reference):
    '''
        Normalize multichannel data with respect to another multichannel data
        each data is of the format : [channel x time]
    '''
    normed = []
    for to_, ref_ in zip(to_normalize_data, normalize_reference):
        normed.append(utils.normalize_energy(to_, ref_))
    return np.array(normed).squeeze().T


class ToSave(object):
    def __init__(self, feats, bunch_id):
        self.feats = feats
        self.id = bunch_id
        self.sir_scale = None
        self.snr_scale = None
        self.src_labels = None
        self.src_phones = None
        self.src_drr = None
        self.src_drr_full = None
        self.src_clean_spec = None
        self.src_early_spec = None
        self.src_reverb_spec = None
        self.src_early_mask = None
        self.src_clean_mask = None
        self.src_reverb_mask = None
        self.inter_labels = None
        self.inter_phones = None
        self.inter_drr = None
        self.inter_drr_full = None
        self.inter_clean_spec = None
        self.inter_reverb_spec = None
        self.inter_mask = None
        self.mixture_spec = None
        self.src_reverb_subseg_spec = None
        self.src_subseg_phones = None
        self.inter_subseg_phones = None

    def save_src_info(self, src_labels, src_phones, src_drr=None,\
            src_clean_spec=None, src_reverb_spec=None, src_early_mask=None):
        self.src_labels = src_labels
        self.src_phones = src_phones
        self.src_drr = src_drr
        self.src_clean_spec = src_clean_spec
        self.src_reverb_spec = src_reverb_spec
        self.src_early_mask = src_early_mask

    def save_inter_info(self, inter_labels, inter_phones, inter_drr=None,\
            inter_clean_spec=None, inter_reverb_spec=None, inter_mask=None):
        self.inter_labels = inter_labels
        self.inter_phones = inter_phones
        self.inter_drr = inter_drr
        self.inter_clean_spec = inter_clean_spec
        self.inter_reverb_spec = inter_reverb_spec
        self.inter_mask = inter_mask

    def save_into_hdf5(self, feats_fid, attr_name, sub_folder, dtype):
        data = getattr(self, attr_name)
        feats_fid.create_dataset(attr_name+'/'+sub_folder, data=data, \
                dtype=dtype, chunks=True, compression='lzf')

def get_uppercase_attr_val(obj, fid=None):
    '''
    returns  a dictionary of the attribute values of a class
    Checks if the attr string is all caps
    if fid is given, writes the attributes into a file
    '''
    ret_val = {}
    for ele in obj.__dir__():
        if ele.isupper():
            ret_val[ele] = obj.__getattribute__(ele)
            if fid is not None:
                fid.write(ele+':'+str(obj.__getattribute__(ele))+'\n')
    return ret_val

def get_all_attr_val(obj, fid=None):
    '''
    returns  a dictionary of the attribute values of a class
    Checks if the attr string is all caps
    if fid is given, writes the attributes into a file
    '''
    ret_val = {}
    for ele in obj.__dir__():
        ret_val[ele] = obj.__getattribute__(ele)
        if fid is not None:
            fid.write(ele+':'+str(obj.__getattribute__(ele))+'\n')
    return ret_val

class NoSpeechWithEnoughSamples(Exception):
    pass

def sec2samples(sec, sampling_rate=config.SAMPLING_RATE):
    return int(sec * sampling_rate)

def compute_phase_features(stft_mic1, stft_mic2):
    '''
    Compute the phase differnce between the stft of the signals
    '''
    angle_diff = np.angle(stft_mic1) - np.angle(stft_mic2)
    cos_theta = np.cos(angle_diff)
    sin_theta = np.sin(angle_diff)
    return np.float32(np.hstack([cos_theta, sin_theta]))


def extract_seg(data, start, end, seg_size, no_buffer=False):
    '''
    Extract segment from data. Data is assumed to be 
    of shape N channels x P samples
    '''
    if no_buffer:
        return data[start:start+seg_size]
    buffer_samples = end - start - seg_size
    assert buffer_samples >= 0, 'Segment size not adequate'
    start = start + int(buffer_samples/2.0)
    end = start + seg_size
    return data[start:end]


def compute_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2)**2))

class SourceManager(object):
    '''
    Definition of a single source in a room.
    '''
    def __init__(self, _id, # speech id
                 speech_file, # speech file
                 rir_obj,
                 rir_file, # RIR between the source and mics
                 speech_align, # speech alignments
                 src_loc, # source location
                 src_distance,
                 src_mic1_dist,
                 src_mic2_dist): # distance between source and mic mid point
        self._id = _id
        self.speech_file = speech_file
        self.rir_file = rir_file
        self.rir_obj = rir_obj
        self.src_distance = src_distance
        self.loc = int(src_loc)
        self.delay = src_distance/config.SOUND_SPEED
        self.delayed_aligns = AlignmentManager(
            readKaldiData.readAlignmentsAsPhoneCTM.\
                        delay_alignment(speech_align, self.delay),
            ignore_list=config.SILENCE_IDS)
        self.mic1_delay = float(src_mic1_dist)/config.SOUND_SPEED
        self.mic2_delay = float(src_mic2_dist)/config.SOUND_SPEED
        self.speech = None
        self.rir_data = None

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


    def reverberate_with_random_noise(self):
        self.read_data()
        speech = self.speech + np.random.rand(len(self.speech))
        return utils.make_multi_channel_reverb(speech, self.rir_data).T

    def reverberate_with_early_echoes(self):
        '''
        Reverberate using the direct path and the early echoes.
        '''
        self.read_data()
        speech = self.speech
        rir_data = self.rir_data
        mic1_sample_delay = int(round(self.mic1_delay*config.SAMPLING_RATE)) + 3
        mic2_sample_delay = int(round(self.mic2_delay*config.SAMPLING_RATE)) + 3
        mic1_sample_len = int(mic1_sample_delay + \
                config.EARLY_ECHOES * config.SAMPLING_RATE)
        mic2_sample_len = int(mic2_sample_delay + \
                config.EARLY_ECHOES * config.SAMPLING_RATE)
        mic1_rir = rir_data[:mic1_sample_len, 0]
        mic2_rir = rir_data[:mic2_sample_len, 1]
        mic1_reverb = utils.make_single_channel_reverb(speech, mic1_rir)
        mic2_reverb = utils.make_single_channel_reverb(speech, mic2_rir)
        return np.array([mic1_reverb, mic2_reverb]).T

    def reverberate_with_direct_sound(self):
        '''
        Reverberate using the direct path and the early echoes.
        '''
        self.read_data()
        speech = self.speech
        rir_data = self.rir_data
        mic1_sample_delay = int(round(self.mic1_delay*config.SAMPLING_RATE)) + 3
        mic2_sample_delay = int(round(self.mic2_delay*config.SAMPLING_RATE)) + 3
        mic1_sample_len = int(mic1_sample_delay)
        mic2_sample_len = int(mic2_sample_delay)
        mic1_rir = rir_data[:mic1_sample_len, 0]
        mic2_rir = rir_data[:mic2_sample_len, 1]
        mic1_reverb = utils.make_single_channel_reverb(speech, mic1_rir)
        mic2_reverb = utils.make_single_channel_reverb(speech, mic2_rir)
        return np.array([mic1_reverb, mic2_reverb]).T

class SeqSourceManager(SourceManager):
    '''
        Class to manage  source along with word alignment
    '''
    def __init__(self, _id, # speech id
                 speech_file, # speech file
                 rir_obj,
                 rir_file, # RIR between the source and mics
                 speech_word_align,
                 speech_align, # speech alignments
                 src_loc, # source location
                 src_distance,
                 src_mic1_dist,
                 src_mic2_dist): # distance between source and mic mid point
        SourceManager.__init__(self, _id, # speech id
                               speech_file, # speech file
                               rir_obj,
                               rir_file, # RIR between the source and mics
                               speech_align, # speech alignments
                               src_loc, # source location
                               src_distance,
                               src_mic1_dist,
                               src_mic2_dist)
        self.delayed_word_aligns = AlignmentManager(
            readKaldiData.readAlignmentsAsPhoneCTM.\
                        delay_alignment(speech_word_align, self.delay),
            seg_dur=config.SEQUENCE_DURATION)

    def get_matched_ph_align(self, word_start, word_dur):
        '''
        Get the phonetic alignments given the word boundaries
        '''
        word_end = word_start + word_dur
        start_idx = None
        end_idx = None
        for idx, align in enumerate(self.delayed_aligns.alignment):
            if abs(align[0] - word_start) <= 1e-4:
                start_idx = idx
            elif abs(align[0] - word_end) <= 1e-4:
                end_idx = idx
                break
        return self.delayed_aligns.alignment[start_idx:end_idx]

    @staticmethod
    def align2ph_frame(phone_alignment, pad_size, \
            segment_size=config.SEQUENCE_DURATION,\
            frame_size=config.FEAT_COMPUTE_DUR,\
            frame_shift=config.SEQUENCE_STFT_SHIFT,\
            sub_frame_size=config.KALDI_FRAME_DUR, \
            sub_frame_shift=config.KALDI_FRAME_SHIFT):
        '''
        create phoneme frames
        creates a representative signal of phone sequence, breaks it into 
        set of frames and then decide the phone id based on the maximum number
        of samples associated to a ph
        '''
        sub_sampling = 1.0
        ph_window_samples = int(sub_frame_size*config.SAMPLING_RATE/sub_sampling)
        ph_shift_samples = int(sub_frame_shift*config.SAMPLING_RATE/sub_sampling)
        ph_sample_seq = []
        max_sample = int(segment_size * config.SAMPLING_RATE/sub_sampling)
        frame_size_sample = int(frame_size * config.SAMPLING_RATE/sub_sampling)
        frame_shift_sample = int(frame_shift* config.SAMPLING_RATE/sub_sampling)
        for ph_align in phone_alignment:
            sample_cnt = int(ph_align[1] * config.SAMPLING_RATE/sub_sampling)
            ph_sample_seq += [ph_align[2]] * sample_cnt
            if len(ph_sample_seq) >= max_sample:
                break
        ph_sample_seq = ph_sample_seq[:max_sample]
        ph_sample_seq = np.pad(ph_sample_seq, pad_size,  mode='reflect')

        frames = librosa.util.frame(np.array(ph_sample_seq),\
                frame_size_sample, frame_shift_sample)
        ph_seq = []
        for sub_frame in frames.T:
            ph_frames = librosa.util.frame(sub_frame,\
                    ph_window_samples, ph_shift_samples)
            ph_sub_frame_seq = []
            for ph_frame in ph_frames.T:
                ph, cnt = np.unique(ph_frame, return_counts=True)
                ph_sub_frame_seq.append(int(ph[np.argmax(cnt)]))
            ph_seq.append(ph_sub_frame_seq)
        return np.array(ph_seq)

    @staticmethod
    def ph_frame2ph_sub_frame(ph_frame, frame_window = \
            int(config.FEAT_COMPUTE_DUR/config.KALDI_FRAME_SHIFT),\
            shift=int(config.SEQUENCE_STFT_SHIFT/config.KALDI_FRAME_SHIFT)):
        return  librosa.util.frame(np.array(ph_frame), frame_window, shift)
        

    @staticmethod
    def compute_sub_spec(wav_data, segment_stft_if, sub_seg_stft_if, \
            window=int(config.FEAT_COMPUTE_DUR * config.SAMPLING_RATE), \
            shift=int(config.SEQUENCE_STFT_SHIFT * config.SAMPLING_RATE)):
        '''
            wav_data is in the format [number_of_channels x time]
            returns [ channel x 100ms frame count x fft_size x 25ms frame count]
        '''
        assert len(wav_data.shape) == 2, \
                'wav data should be of format channel x time'
        spec = []
        for wav in wav_data.T:
            wav = np.pad(wav, int(segment_stft_if.nfft // 2), mode='reflect')
            wav_data_frame = librosa.util.frame(wav, window, shift)
            sub_spec = []
            for sub_wav in wav_data_frame.T:
                sub_wav_stft = np.abs(sub_seg_stft_if.compute_stft(sub_wav))
                sub_spec.append(sub_wav_stft)
            spec.append(np.array(sub_spec))
        return np.array(spec)


class AlignmentManager(object):
    '''
    Class to manage alignment. Used while obtaining overlap region
    between alignments
    Input
    -----
    alignment : a list of  tuple containing (start, duration, ph_id)
    overlap : Amount of overlap needed in secs
    seg_dur : Segment duration to consider
    '''
    def __init__(self, alignment, ignore_list=None, \
            seg_dur=config.MIN_SEGMENT_SIZE_FOR_SEARCH):
        self.alignment = alignment
        self.alignment_len = len(alignment)
        self.next_idx = 0
        if ignore_list is None:
            ignore_list = []
        self.ignore_list = ignore_list
# Positions of start, duration and phone in the alignment tuple
        self.start_pos = 0
        self.duration_pos = 1
        self.phone_pos = 2
        self.eligible_segs = self.get_all_eligible_segments(duration=seg_dur)

    def get_all_eligible_segments(self, \
            duration=config.MIN_SEGMENT_SIZE_FOR_SEARCH):
        '''
            Check for segments who's duration is more
            than MIN_SEGMENT_SIZE_FOR_SEARCH
        '''
        all_segs = []
        while self.next_idx < self.alignment_len:
            if self.alignment[self.next_idx][self.duration_pos] >= duration and\
                    self.alignment[self.next_idx][self.phone_pos] \
                    not in self.ignore_list:
                all_segs.append(self.alignment[self.next_idx])
            self.next_idx += 1
        return all_segs


def create_mask(reverb_data, early_reverb_data, stft_if):
    '''
        Create a mask using reverberation data and early reverb data
        stft interface object determines the the stft dimension
    '''
    total_r_stft = np.zeros((stft_if.stft_freq_dim))
    total_er_stft = np.zeros((stft_if.stft_freq_dim))
    total_l_stft = np.zeros((stft_if.stft_freq_dim))
    for r_data, er_data in zip(reverb_data, early_reverb_data):
        r_stft = stft_if.compute_stft_single_frame(r_data)
        er_stft = stft_if.compute_stft_single_frame(er_data)
        l_stft = r_stft - er_stft
        total_r_stft += np.abs(r_stft)
        total_l_stft += np.abs(l_stft)
        total_er_stft += np.abs(er_stft)
        #mask.append(er_stft/(er_stft + l_stft + np.finfo(float).eps))
    overall_mask = total_er_stft/(total_er_stft + total_l_stft + np.finfo(float).eps)
    #overall_mask = total_er_stft/(total_r_stft + np.finfo(float).eps)
    #return np.array(mask), overall_mask
    return overall_mask

def create_seq_mask(reverb_data, early_reverb_data, stft_if, stft_shape):
    '''
        Create a mask using reverberation data and early reverb data
        stft interface object determines the the stft dimension
    '''
    total_r_stft = np.zeros(stft_shape)
    total_er_stft = np.zeros(stft_shape)
    total_l_stft = np.zeros(stft_shape)
    for r_data, er_data in zip(reverb_data, early_reverb_data):
        r_stft = stft_if.compute_stft(r_data)
        er_stft = stft_if.compute_stft(er_data)
        l_stft = r_stft - er_stft
        total_r_stft += np.abs(r_stft)
        total_l_stft += np.abs(l_stft)
        total_er_stft += np.abs(er_stft)
        #mask.append(er_stft/(er_stft + l_stft + np.finfo(float).eps))
    overall_mask = total_er_stft/(total_er_stft + total_l_stft + np.finfo(float).eps)
    #overall_mask = total_er_stft/(total_r_stft + np.finfo(float).eps)
    #return np.array(mask), overall_mask
    return overall_mask

class FileManager(object):
    '''
    Class to manage speech and rir files
    Input:
    ------
    speech_scp_file : scp file of speech data
    alignment_hdf5 : A hdf5 file containing alignment information
    noise_list : A list of noise files or speech files in case of ssn
        noise generation
    rir_list : A yaml file containing rir data
    noise_type: 'ssn' if noise to be used is speech shaped noise
    '''
    def __init__(self, speech_scp_file, alignment_hdf5, \
            noise_list, rir_list, noise_type):
        self.speech_scp = utils.parse_scp(speech_scp_file)
        self.speech_file_count = len(self.speech_scp)
        self.speech_keys = list(self.speech_scp.keys())
        self.yaml_fid = open(rir_list, 'r')
        self.rir_file = rir_list
        self.yaml_iter = yaml.safe_load_all(self.yaml_fid)
        self.alignment_fid = h5py.File(alignment_hdf5, 'r')
        self.current_idx = 0
        self.id = 0
        self.speech_with_enough_segments = 0
        self.randomize_speech_keys()
        self.noise_list = utils.parse_list(noise_list)
        self.noise_file_cnt = len(self.noise_list)
        self.noise_type = noise_type
        if noise_type == 'ssn':
            seg_size = int(config.SAMPLING_RATE * config.FEAT_COMPUTE_DUR)
            self.ssn_obj = ssn.CreateTwoChannelSSN(seg_size, self.noise_list, \
                    config.DISTANCE_BETWEEN_MICROPHONE, config.SAMPLING_RATE,\
                    shuffle=True)

    def get_noise_data(self, seg_size, reject_file=None):
        if self.noise_type == 'ssn':
            ssn_spec = self.ssn_obj.compute_ssn_spectrum(config.FRAMES_TO_COMPUTE_SSN, \
                    reject_file=reject_file)
            noise = self.ssn_obj.make_ssn(ssn_spec, seg_size)
        else:
            while True:
                noise_file = self.pick_random_noise_file()
                n_info = self.audio_file_info(noise_file)
                noise_len = n_info.frames
                if noise_len < seg_size:
                    continue
                start_pos = np.random.randint(noise_len-seg_size)
                noise = self.read_audio_file(noise_file, start=start_pos, \
                        end=start_pos+seg_size)
                noise = noise[:, :2].T
                break
        return noise

    def reset_rir_fid(self):
        self.yaml_fid.close()
        self.yaml_fid = open(self.rir_file, 'r')
        self.yaml_iter = yaml.safe_load_all(self.yaml_fid)

    def update_id(self):
        self.id += 1

    def pick_random_noise_file(self):
        noise_file_idx = np.random.randint(self.noise_file_cnt)
        return self.noise_list[noise_file_idx]

    def read_audio_file(self, file_name, start=None, end=None):
        if start is None:
            data, _ = sf.read(file_name, always_2d=True)
        else:
            data, _ = sf.read(file_name, always_2d=True, start=start, stop=end)

        return data

    def audio_file_info(self, file_name):
        return sf.info(file_name)

    def __del__(self):
        self.yaml_fid.close()
        self.alignment_fid.close()

    def __read_yaml_data(self):
        ''' Lazy read yaml file'''
        while True:
            rir = next(self.yaml_iter, None)
            if not rir:
                return None
            src_rir_path = rir['rir_base_path']+'/'+\
                    rir['src']['rir_name']
            inter_rir_path = rir['rir_base_path']+'/'+\
                    rir['interference']['rir_name']
            if os.path.exists(src_rir_path) and \
                    os.path.exists(inter_rir_path):
                return rir
            else:
                print('Missing:', rir)

    def update_idx(self):
        self.current_idx += 1
        if self.current_idx >= self.speech_file_count:
            if self.speech_with_enough_segments < 2:
                raise  NoSpeechWithEnoughSamples('No speech signal has ' +\
                    str(config.NUMBER_OF_FRAMES_REQD_PER_FILE) + ' samples')
            self.randomize_speech_keys()
            self.current_idx = 0

    def randomize_speech_keys(self):
        np.random.shuffle(self.speech_keys)


    def get_src_int(self):
        rir = self.__read_yaml_data()
        if not rir:
            return None
        return self.create_src_inter_obj(rir)

    def create_src_inter_obj(self, rir):
        src = self.create_source_obj(rir, 'src')
        inter = self.create_source_obj(rir, 'interference')
        return [src, inter]

    def get_metadata(self, rir, src_type):
        src_pos = rir[src_type]['pos']
        mic1_pos = np.array(rir['mic']['pos'][0])
        mic2_pos = np.array(rir['mic']['pos'][1])
        src_rir_path = rir['rir_base_path']+'/'+ rir[src_type]['rir_name']
        src_loc = rir[src_type]['loc']
        src_mic1_dist = compute_dist(src_pos, mic1_pos)
        src_mic2_dist = compute_dist(src_pos, mic2_pos)
        mid_point = (mic1_pos + mic2_pos)*0.5
        src_dist = compute_dist(src_pos, mid_point)
        #return (src_pos , mic1_pos ,mic2_pos, src_rir_path, src_loc, \
        #        src_mic1_dist, src_mic2_dist , mid_point , src_dist)
        return (src_rir_path, src_loc, src_mic1_dist, src_mic2_dist, src_dist)


    def create_source_obj(self, rir, src_type):
        src_rir_path, src_loc, src_mic1_dist, src_mic2_dist, src_dist = \
                self.get_metadata(rir, src_type)
        while True:
            src_id = self.speech_keys[self.current_idx]
            src_file = self.speech_scp[self.speech_keys[self.current_idx]]
            self.update_idx()
            if src_id in self.alignment_fid['alignments']:
                src_align = self.alignment_fid['alignments'][src_id][...]
                src = SourceManager(src_id, src_file, rir, src_rir_path,\
                        src_align, src_loc, src_dist, src_mic1_dist,\
                        src_mic2_dist)
                if src.delayed_aligns.eligible_segs:
                    self.speech_with_enough_segments += 1
                    return src
            else:
                print('No alignments for:', src_id)

class SeqFileManager(FileManager):
    '''
        File manager to extract features from a word
    '''
    def __init__(self, speech_scp_file, ph_alignment_hdf5, word_align_hdf5, \
            noise_list, rir_list, noise_type):
        FileManager.__init__(self, speech_scp_file, ph_alignment_hdf5, \
                noise_list, rir_list, noise_type)
        self.word_alignment_fid = h5py.File(word_align_hdf5, 'r')

    def create_source_obj(self, rir, src_type):
        src_rir_path, src_loc, src_mic1_dist, src_mic2_dist, src_dist = \
                self.get_metadata(rir, src_type)
        while True:
            src_id = self.speech_keys[self.current_idx]
            src_file = self.speech_scp[self.speech_keys[self.current_idx]]
            self.update_idx()
            if src_id in self.alignment_fid['alignments']:
                ph_align = self.alignment_fid['alignments'][src_id][...]
                wrd_align = self.word_alignment_fid['alignments'][src_id][...]
                src = SeqSourceManager(src_id, src_file, rir, src_rir_path,\
                        wrd_align, ph_align, src_loc, src_dist, src_mic1_dist,\
                        src_mic2_dist)
                if src.delayed_word_aligns.eligible_segs:
                    self.speech_with_enough_segments += 1
                    return src
            else:
                print('No alignments for:', src_id)


    def __del__(self):
        FileManager.__del__(self)
        self.word_alignment_fid.close()

def save_sub_feat(feats_fid, data, label, dtype):
    feats_fid.create_dataset(label, data=data, dtype=dtype, chunks=True, \
            compression='lzf')

