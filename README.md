# PyAnnote Wrapper for Speaker Diarization

## Description
This app is used to do speaker diarizaiton on audio and video files. This is a wrapper of 
[PyAnnote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements
* Requires Python3 with `clams-python`, `clams-utils`, `torch` and `pyannote.audio` to run the app locally.
* Requires an HTTP client utility (such as `curl`) to invoke and execute analysis.
* Requires docker to run the app in a Docker container 

Run `pip install -r requirements.txt` to install the requirements.

### Speaker diarization for a file 
`PyAnnote` requires users to agree with 2 user conditions to get access to their model on HuggingFace. 
Therefore, the user of this app must have satisfied this requirement with their own HuggingFace account. 
After agreeing with the conditions, to run this app in CLI, copy the access token and run:

`python cli.py --huggingface_token <huggingface_access_token> <input_mmif_file_path> <output_mmif_file_path>`


### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) 
or the [`metadata.py`](metadata.py) file in this repository.
