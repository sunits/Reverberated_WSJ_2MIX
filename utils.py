import olafilt
import numpy as np


def make_multi_channel_reverb(data, rir):
    '''
    Apply RIR on to a signal to obtain reverberated
    speech.
    '''
    reverb_data = []
    for rir_ in rir.T:
        reverb_data.append(make_single_channel_reverb(data, rir_))
    return np.array(reverb_data)

def make_single_channel_reverb(data, rir):
    return olafilt.olafilt(rir, data)

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

def parse_list(filelist):
    '''
    Parse a list containing files
    '''
    files = []
    with open(filelist) as fid:
        for file_ in fid:
            files.append(file_.strip())
    return files

def open_rir_yaml_file(fid):
    '''
    Read the yaml file
    '''
    #rir_info = list(yaml.load_all(fid, Loader=yaml.CLoader))
    rir_info = yaml.load_all(fid)
    #print('Number of RIRs:'+ str(len(rir_info)))
    return rir_info

def parse_scp(filelist):
    '''
    Parse a list containing files
    '''
    files = {}
    with open(filelist) as fid:
        for file_ in fid:
            id_, path = file_.strip().split()
            files[id_] = path
    return files

def normalize_energy(data, normalize_to_sig):
    '''
      Ensure that the energy of data is same as normalize_to_sig
    '''
    if len(data.shape) == 1:
        data = data.reshape(1, len(data))
    normalize_signal = np.zeros((data.shape[0], data.shape[1]))
    for idx, sig in enumerate(data):
        normalize_signal[idx, :] = np.sqrt(np.sum(normalize_to_sig**2)/np.sum(sig**2)) * sig
    return normalize_signal
