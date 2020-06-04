# meta-metadata

This README.md describes about each metadata present in this folder.

- Metadata for LibriTTS train-clean-100 split 
  - `libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist_22k.txt`
  - `libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist_22k.txt`
  - These metadata are from [NVIDIA/mellotron](https://github.com/NVIDIA/mellotron).
- Metadata for VCTK corpus
  - `vctk_22k_train.txt`: List of all training files from VCTK
  - `vctk_22k_train_10s.txt`: List of training files that are shorter than 10 seconds
  - `vctk_22k_val.txt`: val.
  - `vctk_22k_test.txt`: test.
  - For details of making train/val/test split, see section 3.1 of our [paper](https://arxiv.org/abs/2005.03295).
- `libritts_vctk_speaker_list.txt`
  - This is a list of all speakers that are present in LibriTTS train-clean-100 and VCTK.
  - You may want to copy-paste the line below in "speakers" of global configuration yaml file.
