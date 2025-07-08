# context-dependency-resolution 🔍

In this project, I compare different methods to solve the task of Context Dependency Resolution, more commonly known as **Incomplete Utterances Rewriting (IUR)**. This task involves rewriting ambiguous parts of sentences (e.g., pronouns, ellipsis) based on previous dialogue turns.


## Content 📖
- [Stack of Technologies](#stack-of-technologies)
- [Task Description](#task-description)
- [Exploratory Data Analysis](#eda)
- [Proposed Solution](#proposed-solution)
- [Main Results](#main-results)
- [Future Research](#future-research)

## Stack of Technologies 🏗️
- Programming language:
  - Python 🐍
- Data handling:
  - pandas
  - datasets
- Model fine-tuning:
  - Transformers 🤗
  - peft
  - trl 
- MLOps, experiments logging:
  - Weights & Biases (WandB) 🪄
  - omegaconf
  - yaml
- Evaluation:
  - evaluate
- Visualization:
  - matplotlib
  - seaborn

## Task Description 📋

**Goal**: Disambiguate the final utterance based on dialogue context  
**Practical Value**: Pipeline component for dialogue systems (e.g., text-to-SQL bots for analytics)

**Example Dialogue**:

👤 User: Сколько литров молока было продано 1 апреля в Ульяновск?  
🤖 Assistant: 1 апреля в Ульяновске было продано 1000 литров молока  
👤 User: А сколько пачек масла?  
🔧 **Rewritten User Query**: А сколько пачек масла было продано 1 апреля в Ульяновске?  

## EDA 📊
number of dialogs: 1048
number of phrases: 5514

**Length distribution of dialog history**

         in characters     in words
mean      363.195140
std       302.450948
min         7.000000
25%       132.000000
50%       299.000000
75%       506.000000
max      2605.000000


**Length distribution of initial and rewritten messages**

![image](https://github.com/user-attachments/assets/5461aab5-63e6-423b-8082-18613b5666aa)


**PoS distribution of tokens in initial and rewritten messages**

![image](https://github.com/user-attachments/assets/e00d4219-d73a-4749-88cc-48e0a8b54962)



## Proposed Solution 🚀

Comparison between API-based LLM prompting vs. local model fine-tuning:

| **Approach** | **Pros** | **Cons** |
|--------------|----------|----------|
| **Prompting** | • Minimal data requirements<br>• Low computational demand | • Provider dependency<br>• Higher costs<br>• Less customizable |
| **Fine-tuning** | • No provider dependency<br>• Highly customizable<br>• Cost-effective long-term | • Requires labeled data<br>• Demands compute resources |

**Evaluated Models**:
- Prompting (zero/few-shot):  
  `Deepseek_R1`, `Deepseek_V3`, `gpt-4o-mini`, `Llama-3.3-70B-Instruct `
- Fine-tuning (LoRA, QLoRA, p-tuning):  
  `ruT5-base`, `ruT5-large`, `Vikhr-Gemma-2B-instruct`

## Main Results 💡
**Evaluation metrics**:
- BLEU
- ROUGE
- Resroration-score (rf-score)

$$
precision_n = \frac{|\{restored\ n\\_grams\} \cap \{n\\_grams\ in\ ref\}|}{|\{restored\ n\\_grams\}|}
$$

$$
recall_n = \frac{|\{restored\ n\\_grams\} \cap \{n\\_grams\ in\ ref\}|}{|\{n\\_grams\ in\ ref\}|}
$$

$$
f1_n = 2 * \frac{precision_n * recall_n}{precision_n + recall_n}
$$


**Experiments with prompting**

| Metrica       | Llama-3.3-70B-Instruct |      Llama-3.3-70B-Instruct    | gpt-4o-mini    |    gpt-4o-mini   | DeepSeek-V3    |   DeepSeek-V3       | DeepSeek-R1    | DeepSeek-R1 |
|-----------------|------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|  | **Zero-shot**         | **Few-shot**   | **Zero-shot**  | **Few-shot**   | **Zero-shot**  | **Few-shot**   | **Zero-shot**  | **Few-shot**   |
| bleu_score      | 46.45                 | 51.33          | 63.16          | **63.63**      | 59.86          | 37.34          | 35.08          | 41.76          |
| rouge-1         | 0.575                 | 0.586          | 0.622          | **0.643**      | 0.617          | 0.484          | 0.556          | 0.561          |
| rouge-2         | 0.445                 | 0.436          | 0.489          | 0.488          | **0.496**      | 0.315          | 0.382          | 0.374          |
| rouge-3         | 0.354                 | 0.339          | 0.400          | 0.388          | **0.419**      | 0.217          | 0.274          | 0.254          |
| rouge-4         | 0.281                 | 0.264          | **0.330**      | 0.308          | 0.354          | 0.145          | 0.192          | 0.165          |
| rf_score_1      | 0.234                 | 0.342          | 0.277          | **0.396**      | 0.296          | 0.349          | 0.251          | 0.357          |
| rf_score_2      | 0.165                 | 0.249          | 0.202          | **0.304**      | 0.233          | 0.255          | 0.186          | 0.272          |
| rf_score_3      | 0.135                 | 0.201          | 0.174          | **0.268**      | 0.204          | 0.212          | 0.157          | 0.242          |
| rf_score_4      | 0.119                 | 0.173          | 0.159          | **0.247**      | 0.187          | 0.187          | 0.135          | 0.214          |

**Experiments with fine-tuning**

*rut5-base fine-tuning*

<img width="319" alt="image" src="https://github.com/user-attachments/assets/f56b6140-1dd0-42c0-8f5f-a2154a4e43b1" />

*rut5-large fine-tuning*

<img width="468" alt="image" src="https://github.com/user-attachments/assets/260631b9-987e-48d5-b42a-082e95f95bff" />

*Vikhr-Gemma-2B-instruct fine-tuning (p-tuning and QLoRA)*
<img width="712" alt="image" src="https://github.com/user-attachments/assets/64e801d8-e57f-4766-9baf-8af9f78315e0" />

| Название модели               | rut5-base       | rut5-large (1) | rut5-large (2) | rut5-large (3) | rut5-large (4) |
|-------------------------------|-----------------|----------------|----------------|----------------|----------------|
| Способ дообучения             | Полный файнтонит | LoRA файнтонит | LoRA файнтонит | LoRA файнтонит | LoRA файнтонит |
| Количество параметров         | 222 903 552     | 742 386 688    | 742 386 688    | 742 386 688    | 742 386 688    |
| Количество обучаемых параметров | 222 903 552     | 1 179 648      | 4 718 592      | 4 718 592      | 4 718 592      |
| bleu_score                    | 73.34           | 73.33          | **73.87**      | 73.17          | 73.72          |
| rouge-1                       | 0.734           | 0.740          | 0.732          | **0.734**      | 0.742          |
| rouge-2                       | 0.618           | 0.643          | 0.628          | 0.625          | 0.634          |
| rouge-3                       | 0.530           | **0.565**      | 0.551          | 0.539          | 0.551          |
| rouge-4                       | 0.465           | **0.493**      | 0.482          | 0.471          | 0.476          |
| rf_score_1                    | 0.407           | 0.285          | 0.338          | **0.431**      | 0.401          |
| rf_score_2                    | 0.324           | 0.215          | 0.260          | **0.337**      | 0.323          |
| rf_score_3                    | 0.289           | 0.188          | 0.230          | **0.303**      | 0.292          |
| rf_score_4                    | 0.270           | 0.176          | 0.213          | **0.282**      | 0.274          |

## Future Research 🔨
Exploring approaches that frame IUR as edit matrix prediction rather than seq2seq. Key benefits include faster parallel operations versus autoregressive generation.

**References**:  
1. [Incomplete Utterance Rewriting as Semantic Segmentation](https://arxiv.org/abs/2009.13166)  
   <img width="1234" alt="Edit Matrix Diagram" src="https://github.com/user-attachments/assets/91440d0a-9b83-49f3-8b9a-f8c8a7653304" />  
2. [How Well Apply Simple MLP to Incomplete Utterance Rewriting?](https://aclanthology.org/2023.acl-short.134)  
   <img width="1118" alt="image" src="https://github.com/user-attachments/assets/bd0a04fc-27f4-4b06-abfc-bc72210d0c08" />

## Contacts 📞
https://t.me/pa-shk

https://www.linkedin.com/in/pa-shk

pvlshknv@gmail.com
