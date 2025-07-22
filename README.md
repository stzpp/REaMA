# REaMA: Building Biomedical Relation Extraction Specialized Large Language Models Through Instruction Tuning


We introduce REaMA, a series of open-source LLMs with sizes of 7B and 13B specifically tailored for biomedical relation extraction tasks. REaMA is fine-tuned on LLaMA-2 series model. 


# Dataset
- The pre-processed training dataset can be downloaded at https://github.com/stzpp/REaMA/blob/main/data.zip
- The test dataset can be downloaded at https://github.com/stzpp/REaMA/blob/main/data.zip

# REaMA
- The fine-tuned REaMA 7B can be downloaded at git clone https://www.modelscope.cn/zppqiwen/REaMA-7B.git
- The fine-tuned REaMA 13B can be downloaded at git clone https://www.modelscope.cn/zppqiwen/REaMA-13B.git
- The fine-tuned REaMA-13B-CoT fine-tuning using cot response can be downloaded at git clone https://www.modelscope.cn/zppqiwen/REaMA-CoT.git

# Requirements
- transformers>=4.31.0 
- torch>=2.1.0
- vllm>=054

# Usage

Reproduce the fine-tuned models
```python
bash train.sh
```

Reproduce the evaluation results
```python
bash test.sh
```


Start chatting with `REaMA` using the following code snippet:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

prompts = []
prompt = "<|user|>\n" + your_query + "\n<|assistant|>\n" 
prompts.append(prompt)
tp_size = 4   
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop = ["\nInput", "USER:", "USER", "ASSISTANT:", "ASSISTANT"])
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.8, tensor_parallel_size=tp_size)
outputs = llm.generate(prompts, sampling_params)
generated_text = [output.outputs[0].text for output in outputs]
outputs = generated_text[:]

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

REaMA should be used with this prompt format:
```
### User:
Your query here

### Assistant
The output of REaMA
```


## Limitations

### Intended Use

These models are intended for research only, in adherence with the [CC BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.


## Citations

```bibtext
@misc{touvron2023llama,
      title={Llama 2: Open Foundation and Fine-Tuned Chat Models}, 
      author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
      year={2023},
      eprint={2307.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

