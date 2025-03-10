# Flow Matching Text-To-Music Tutorial

This is a PyTorch tutorial on Flow Matching for Text-To-Music. 
The main goal of this repository is to learn flow matching at the code level through a fun task and a simple dataset.

## Setup
### Clone the Repository
```
git clone git@github.com:jakeoneijk/FlowMatchingTextToMusicTutorial.git
```
```
cd FlowMatchingTextToMusicTutorial
```
### Create a Conda Environment (Optional)
If you don't want to use a Conda environment, you may skip this step.
```
source conda create -n flow python==3.11
```
```
conda activate flow
```
### Install [PyTorch](https://pytorch.org/get-started/locally/). 
👉 You should check your CUDA Version and install compatible version.

### Install Requirements
```
pip install -r ./requirements.txt
```

### Download Pretrained Weights
Download the pretrained weights for both the **AutoEncoder** and **CLAP** models:
- [AutoEncoder](https://huggingface.co/datasets/jakeoneijk/FlowMatchingTextToMusicTutorial/tree/main)
- [CLAP](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt)

Save them to the following directory:
 ```
  .
  └── CKPT
      ├── autoencoder.pth
      └── music_audioset_epoch_15_esc_90.14.pt
  ```
### Download [Medley-solos-DB](https://zenodo.org/records/3464194)
Download the Medley-solos-DB dataset and place it in the following directory:
 ```
  .
  └── Data
      └── Dataset
          └── MedleySolosDB
              ├── ~.wav
              ├── ...
              └── ~.wav
  ```
## Training
### Check ```HParams.py``` for Configurations
```python
class Mode:
    # You can choose how to optimize the model
    config_name:str = [
        'diffusion', 
        'flow'
    ][1]
    # Currently only supports the "train" stage  
    stage:str = {
        0:"preprocess", 
        1:"train", 
        2:"inference", 
        3:"evaluate"
    }[1]

class Resource:
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
### Train the Model
If you don’t set ```lv``` (log visualizer), TensorBoard will be used by default.
```
python Main.py -lv wandb -do
```

## References
- [StableAudioOpen](https://github.com/Stability-AI/stable-audio-tools)