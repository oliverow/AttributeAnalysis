## Installation
```
conda create -n attributeanalysis python=3.9
conda activate attributeanalysis
```

## Pretrained weights
- [dlib landmarks detector](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) 
- [FFHQ e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing)
- [FFHQ stylegan2-ada](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) 
- [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)

Download and save this pretrained weights in `pretrained/` directory

## Run a demo for attribute anaylsis
```
python attribute_analysis.py
```

## References
1. [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
2. [CLIP](https://github.com/openai/CLIP.git)
3. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
4. [StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch)