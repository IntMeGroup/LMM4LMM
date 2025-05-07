
<div align="center">
   <h1>LMM4LMM: Benchmarking and Evaluating Large-multimodal Image Generation with LMMs</h1>
   <i>How to evaluate Text to Image Generation Model properly?</i>
   <div>
      <!-- <a href="https://arxiv.org/abs/2504.08358"><img src="https://arxiv.org/abs/2504.08358"/></a> -->
      <a href="https://arxiv.org/abs/2504.08358"><img src="https://img.shields.io/badge/Arxiv-2504.08358-red"/></a>
      <a href="https://huggingface.co/datasets/wangjiarui/EvalMi-50K/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green"></a>
   </div>
</div>

![pipeline](https://github.com/user-attachments/assets/c322826f-12f3-48a1-b62f-b1c731dc4ba6)
## T2V Model Ranks
![modelrank](https://github.com/user-attachments/assets/1ff75fa2-f9fe-43c1-8e34-bd72d9a9d443)
## LMM-T2V Models
![Models](https://github.com/user-attachments/assets/fad370b3-9a65-4625-8542-03e11550c335)
## LMM-VQA Models
![LMMs](https://github.com/user-attachments/assets/dea7d25a-1ba3-4865-b4d5-ebf6857842c3)
## EvalMi-50K Download
 <a href="https://huggingface.co/datasets/wangjiarui/EvalMi-50K/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green"></a>
```
huggingface-cli download wangjiarui/EvalMi-50K --repo-type dataset --local-dir ./EvalMi-50K
```
## 🛠️ Installation

Clone this repository:
```
git clone https://github.com/IntMeGroup/LMM4LMM.git
```
Create a conda virtual environment and activate it:
```
conda create -n LMM4LMM python=3.9 -y
conda activate LMM4LMM
```
Install dependencies using requirements.txt:
```
pip install -r requirements.txt
```

## 🌈 Training

for stage1 training (Text-based quality levels)

```
sh shell/train_stage1.sh
```
for stage2 training (Fine-tuning the vision encoder and LLM with LoRA)

```
sh shell/train_stage2.sh
```

for quastion-answering training (QA)
```
sh shell/train_qa.sh
```

## 🌈 Evaluation
Download the pretrained weights
```
huggingface-cli download IntMeGroup/LMM4LMM-Perception --local-dir ./weights/stage2/stage2_mos1
huggingface-cli download IntMeGroup/LMM4LMM-Correspondence --local-dir ./weights/stage2/stage2_mos2
huggingface-cli download IntMeGroup/LMM4LMM-QA --local-dir ./weights/qa
```

for perception and correspondence score evaluation (Scores)

```
sh shell/eval_scores.sh
```

for quastion-answering evaluation (QA)
```
sh shell/eval_qa.sh
```


## 📌 TODO
- ✅ Release the training code 
- ✅ Release the evaluation code 
- ✅ Release the AIGVQA-DB

## Quick Access of T2V Models
| Model |Code/Project Link |
|---|---|
|SD_v2-1|https://huggingface.co/stabilityai/stable-diffusion-2-1|
|i-Code-V3|https://github.com/microsoft/i-Code|
|SDXL_base_1|https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0|
|DALLE3|https://openai.com/index/dall-e-3|
|LLMGA|https://github.com/dvlab-research/LLMGA|
|Kandinsky-3|https://github.com/ai-forever/Kandinsky-3|
|LWM|https://github.com/LargeWorldModel/LWM|
|Playground|https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic|
|LaVi-Bridge|https://github.com/ShihaoZhaoZSH/LaVi-Bridge|
|ELLA|https://github.com/TencentQQGYLab/ELLA|
|Seed-xi|https://github.com/AILab-CVC/SEED-X|
|PixArt-sigma|https://github.com/PixArt-alpha/PixArt-sigma|
|LlamaGen|https://github.com/FoundationVision/LlamaGen|
|Kolors|https://github.com/Kwai-Kolors/Kolors|
|Flux_schnell|https://huggingface.co/black-forest-labs/FLUX.1-schnell|
|Omnigen|https://github.com/VectorSpaceLab/OmniGen|
|EMU3|https://github.com/baaivision/Emu|
|Vila-u|https://github.com/mit-han-lab/vila-u|
|SD3_5_large|https://huggingface.co/stabilityai/stable-diffusion-3.5-large|
|Show-o|https://github.com/showlab/Show-o|
|Janus|https://github.com/deepseek-ai/Janus|
|Hart|https://github.com/mit-han-lab/hart|
|NOVA|https://github.com/baaivision/NOVA|
|Infinity|https://github.com/FoundationVision/Infinity|

## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `wangjiarui@sjtu.edu.cn`


## 🎓Citations

If you find our work useful, please cite our paper as:
```
@misc{wang2025lmm4lmmbenchmarkingevaluatinglargemultimodal,
      title={LMM4LMM: Benchmarking and Evaluating Large-multimodal Image Generation with LMMs}, 
      author={Jiarui Wang and Huiyu Duan and Yu Zhao and Juntong Wang and Guangtao Zhai and Xiongkuo Min},
      year={2025},
      eprint={2504.08358},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.08358}, 
}
```
