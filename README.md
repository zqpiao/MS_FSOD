# MS_FSOD
Enhancing Few-Shot Object Detection through Mixing and Separating Tuning Strategies
## How to reproduce Mixing_Separating


Following the original implementation, it consists of 2 steps:

- **Step1: Mixing**:
   - use the public available pre-trained model on base datasets as model initialization and further fine tune the bbox head with mixing few shot finetuning datasets.

- **Step2: Separating**:
   - use the mixing model from step1 as model initialization and further fine tune the bbox head with few shot datasets.
