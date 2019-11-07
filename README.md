# Kinect-WSJ
Code to simulate a reverberated, noisy version of the WSJ0-2MIX dataset. Microphones are placed on a linear array with spacing between the devices resembling that of Microsoft Kinect &trade;, the device used to record the CHiME-5 dataset. This was done so that we could use the real ambient noise captured as part of CHiME-5 dataset. The room impulse responses (RIR) were simulated for a sampling rate of 16,000 Hz.


## Requirements
* [CHIME5 data set](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5)
* [Dihard 2 labels](https://coml.lscp.ens.fr/dihard/index.html)
* [WSJ 2 mix](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip)

## Instructions

Run ``` ./create_corrupted_speech.sh --stage 0 --wsj_data_path  wsj_path --chime5_wav_base chime_path --dihard_sad_label_path dihard_path --dest save_path```
* wsj_path :  Path to precomputed wsj-2mix dataset. Should contain the folder 2speakers/wav16k/
* chime_path : Path to chime-5 dataset. Should contain the folders train, dev and eval
* dihard_path : Path to dihard labels. Should contain ```*.lab``` files for the train and dev set


## Output Data
Creates the following sub-folders in each of tr, tt and cv folders:

* s1 : spatial image of s1 (Reverberated version of s1 speaker)
* s2 : spatial image of s2 (Reverberated version of s2 speaker)
* s1_direct :  direct component of s1 at each of the microphones
* s2_direct : direct component of s2 at each of the microphones
* s1_early : Contains only the early reflections (first 50 ms. see config.py to change the value) of s1 at each of the microphones
* s2_early : Contains only the early reflections (first 50 ms. see config.py to change the value) of s2 at each of the microphones
* noise : Contains the noise imposed for each mixture
* mix : s1 + s2 + noise
* list.yaml : A yaml file containing the positions and direction of arrival (DOA) for each utterance and speaker


## References

[Analyzing the impact of speaker localization errors on speech separation for automatic speech recognition](https://arxiv.org/abs/1910.11114)
