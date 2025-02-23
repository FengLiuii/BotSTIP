# From the Past to the Present: A Social Bot Detection Method Based on Spatio-temporal Interactive Perception  --> BotSTIP
Implementation for the paper "From the Past to the Present: A Social Bot Detection Method Based on Spatio-temporal Interactive Perception".

BotSTIP is a framework that leverages the HyperGraph and Transformer for social bot detection.

## Requirements
* torch-geometric==2.1.0
* torch==1.13.0

## data struct

![image](https://github.com/user-attachments/assets/5368717b-2232-491d-829b-c35070a0b86b)


## Train & Test

python main.py --dataset_name "Twibot-20" --batch_size 64 --hidden_dim 128 --weight_decay 1e-2 --structural_learning_rate 1e-4 --temporal_learning_rate 1e-5  --early_stop --patience 20

### Dataset Preparation
The original datasets are available at [Twibot-20](https://github.com/BunsenFeng/TwiBot-20) and [Twibot-22](https://github.com/LuoUndergradXJTU/TwiBot-22). 

