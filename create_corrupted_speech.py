'''
    Create a reverberated and noisy audio
'''

import os
import argparse
import yaml
import soundfile as sf
import numpy as np
import wsj_utils 
import config
#import ipdb

def normalize_multichannel_preserve_ild(to_normalize_data, normalize_reference):
    '''
    use the same scaling factor for all channels
    '''
    ref_energy = np.sum(normalize_reference**2)
    sig_energy = np.sum(to_normalize_data[0]**2)
    norm_factor = np.sqrt(ref_energy/sig_energy)
    return norm_factor * to_normalize_data, norm_factor

def round_to_n_decimals(val, n_decimals):
    '''
        Round float to n decimals
    '''
    val = int(val * 10**n_decimals)/float(10**n_decimals)
    return val

def scale_from_sxr(sxr):
    '''
    #20log(speech/noise) = sxr
    scale = speech/noise
    '''
    return round_to_n_decimals(np.power(10, sxr/20.0), 2)

class SaveContainer(object):
    def __init__(self, wav_id, save_destination, src_count, rir_obj):
        self.save_path = save_destination
        self.reverberated_mix = None
        self.reverberated_source = None
        self.direct_source = None
        self.early_sources = None
        self.noise = None
        self.mic_pos = None
        self.source_pos = None
        self.wav_id = wav_id
        self.src_count = src_count
        self.rir_obj = rir_obj

    def save(self):
        self.mkdir()
        reverb_path = self.save_path + '/mix/' + self.wav_id
        sf.write(reverb_path, self.reverberated_mix, 16000)
        noise_path = self.save_path + '/noise/' + self.wav_id
        sf.write(noise_path, self.noise, 16000)
        src_paths = []
        for idx in range(self.src_count):
            new_src_path = self.save_path + '/s' + str(idx+1) + '/' + self.wav_id
            sf.write(new_src_path, self.reverberated_source[idx], 16000)

            src_direct_path = self.save_path + '/s' + str(idx+1) + '_direct/' + self.wav_id
            sf.write(src_direct_path, self.direct_source[idx].T, 16000)

            src_early_path = self.save_path + '/s' + str(idx+1) + '_early/' + self.wav_id
            sf.write(src_early_path, self.early_sources[idx].T, 16000)
        entry = self.get_mic_src_doa()
        #entry['base_path'] = self.save_path
        entry['wav_id'] = self.wav_id
        entry['src'] = self.rir_obj['src']

    def mkdir(self):
        reverb_path = self.save_path + '/mix' 
        if not os.path.exists(reverb_path):
            os.system('mkdir -p '+reverb_path)
        noise_path = self.save_path + '/noise' 
        if not os.path.exists(noise_path):
            os.system('mkdir -p '+noise_path)
        for idx in range(self.src_count):
            new_src_path = self.save_path + '/s' + str(idx+1)
            early_src_path = self.save_path + '/s' + str(idx+1) + '_early'
            direct_src_path = self.save_path + '/s' + str(idx+1) + '_direct'
            if not os.path.exists(new_src_path):
                os.system('mkdir -p '+new_src_path)
                os.system('mkdir -p '+early_src_path)
                os.system('mkdir -p '+direct_src_path)

    def get_mic_src_doa(self):
        entry = {}
        entry['doa'] = {}
        for src_idx in range(self.src_count):
            entry['doa'][src_idx+1] = self.rir_obj['src'][src_idx]['loc']
        return entry
        



def create_noisy_speech(sources, noise, snr, save_container):
    reverb_sources = []
    orig_sources = []
    anechoic_sources = []
    early_sources = []
    snr_scale = scale_from_sxr(np.random.rand() * snr)
    for src in sources:
        reverb_sources.append(src.reverberate().T)
        orig_sources.append(src.speech)
        anechoic_sources.append(src.reverberate_with_direct_sound().T)
        early_sources.append(src.reverberate_with_early_echoes().T)
    scaled_reverb, scale_factors = scale_sources(reverb_sources, \
            orig_sources, anechoic_sources)
    early_sources = np.einsum('i...,i->i...',np.stack(early_sources, 0), scale_factors)
    anechoic_sources = np.einsum('i...,i->i...',np.stack(anechoic_sources, 0), scale_factors)
    mixed_speech = scaled_reverb.sum(axis=0)
    noise_normed, _ = normalize_multichannel_preserve_ild(noise, scaled_reverb[0][0])
    noise_mixed_speech = snr_scale * mixed_speech + noise_normed
    normed_noise_mixed_speech, scaling_factor  = \
            normalize_multichannel_preserve_ild(noise_mixed_speech, \
            scaled_reverb[0][0])
    save_container.reverberated_mix = normed_noise_mixed_speech.T
    save_container.reverberated_source = scaling_factor * snr_scale* np.swapaxes(scaled_reverb, 1, 2)
    save_container.noise = scaling_factor * noise_normed.T
    save_container.direct_source = snr_scale * scaling_factor * anechoic_sources
    save_container.early_sources = snr_scale * scaling_factor * early_sources

def scale_sources(reverb_source, orig_sources , anechoic_sources):
    '''
    scale to original snr
    '''
    orig_sig = orig_sources[0]
    ref_anechoic = anechoic_sources[0]
    scaled_rev = [reverb_source[0]]
    scale_factors = [1]
    for _rev_, _orig_, _anechoic_ in zip(reverb_source[1:], orig_sources[1:],\
            anechoic_sources[1:]):
        orig_ratio = np.sqrt(np.sum(_orig_ ** 2)/np.sum(orig_sig**2))
        anechoic_ratio = np.sqrt(np.sum(ref_anechoic**2)/np.sum(_anechoic_**2))
        scaling_factor = orig_ratio * anechoic_ratio
        scale_factors.append(scaling_factor)
        scaled_rev.append(scaling_factor * _rev_)
    return np.stack(scaled_rev, axis=0), np.array(scale_factors)


def save_wav_file(data, src, dest_path, src_break_at):
    dest_file = dest_path + '/' + src.speech_file.split(src_break_at)[-1]+'.wav'
    os.system('mkdir -p '+os.path.dirname(dest_file))
    sf.write(dest_file, data, config.SAMPLING_RATE)


def main():
    parser = argparse.ArgumentParser('Create reverberated speech signals')
    parser.add_argument('wsj_mix_list', help='Information containing wav ids and scale ratio \
            such as 011a0109_1.6622_01oo0306_-1.6622.wav')
    parser.add_argument('wsj_mix_list_start', type=int, help='start from index in wsj_mix_list')
    parser.add_argument('wsj_mix_list_end', type=int, help='end at index in wsj_mix_list')
    parser.add_argument('wav_base_path', help='Base path containing the wsj0-mix files')
    parser.add_argument('noise_list', help='A list containing (REVERB) noise or list of speech  \
            sentences for ssn noise')
    parser.add_argument('src_count', type=int, help='Number of sources: 2 or 3')
    parser.add_argument('rir_yaml_list', help='A yaml file containing RIR information')
    parser.add_argument('SNR', type=float, help='Maximum signal to noise ratio')
    parser.add_argument('dest_base', help='destination base path')
    parser.add_argument('--noise_type', type=str, required=False, default='', \
            help='Type of noise to be used. Example: ssn for \
            speech shaped noise. Default to noise from noise_list', )
    args = parser.parse_args()

    if not os.path.exists(args.dest_base):
        os.system('mkdir -p '+args.dest_base)
    fm_if = wsj_utils.WsjFileManager(args.wsj_mix_list, args.wav_base_path, \
            args.src_count, args.noise_list, args.rir_yaml_list, \
            args.wsj_mix_list_start, args.wsj_mix_list_end)
    #ipdb.set_trace()
    while True:
        sources = fm_if.create_sources()
        if sources is None:
            break
        print(sources[0]._id)
        save_container  = SaveContainer(sources[0]._id, args.dest_base,\
                args.src_count, sources[0].rir_obj)
        sources[0].read_data()
        noise = fm_if.get_noise_data(len(sources[0].speech))
        create_noisy_speech(sources, noise, args.SNR, save_container)
        save_container.save()


if __name__ == '__main__':
    main()
