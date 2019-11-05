# Reverberated_WSJ_2MIX
Code to simulate a reverberated, noisy version of the WSJ-2MIX dataset. 


## Requirements
* [CHIME5 data set](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5)
* [Dihard 2 labels](https://coml.lscp.ens.fr/dihard/index.html)
* [WSJ 2 mix](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip)

## Instructions

Run ``` ./create_corrupted_speech.sh```

## Output Data
Creates the following sub-folders in each of tr, tt and cv folders:

* s1 : spatial image of s1 (Reverberated version of the s1)
* s2 : spatial image of s2 (Reverberated version of the s2)
* s1_direct :  direct component of s1 at each of the microphones
* s2_direct : direct component of s2 at each of the microphones
* s1_early : Contains only the early reflections (first 50 ms. see config.py to change the value) of s1 at each of the microphones
* s2_early : Contains only the early reflections (first 50 ms. see config.py to change the value) of s2 at each of the microphones
* noise : Contains the noise imposed for each mixture
* mix : s1 + s2 + noise
* list.yaml : A yaml file containing the positions and direction of arrival (DOA) for each utterance and speaker

## References

[Analyzing the impact of speaker localization errors on speech separation for automatic speech recognition](https://arxiv.org/abs/1910.11114)
