# context-dependency-resolution 🔍

In this project, I compare different methods to solve the task of Context Dependency Resolution, more commonly known as **Incomplete Utterances Rewriting (IUR)**. This task involves rewriting ambiguous parts of sentences (e.g., pronouns, ellipsis) based on previous dialogue turns.

## Content 📖

- [Stack of Technologies](#stack-of-technologies)
- [Task Description](#task-description)
- [Exploratory Data Analysis](#eda)
- [Proposed Solution](#proposed-solution)
- [Main Results](#main-results)
- [Future Research](#future-research)
- [Contact](#feel-free-to-contact-me)

## Stack of Technologies 🏗️

- Python 🐍
- Transformers 🤗
- Weights & Biases (WandB) 🪄

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
- Fine-tuning:  
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


## Future Research 🔨
Exploring approaches that frame IUR as edit matrix prediction rather than seq2seq. Key benefits include faster parallel operations versus autoregressive generation.

**References**:  
1. [Incomplete Utterance Rewriting as Semantic Segmentation](https://arxiv.org/abs/2009.13166)  
   <img width="1234" alt="Edit Matrix Diagram" src="https://github.com/user-attachments/assets/91440d0a-9b83-49f3-8b9a-f8c8a7653304" />  
2. [How Well Apply Simple MLP to Incomplete Utterance Rewriting?](https://aclanthology.org/2023.acl-short.134)  
   <img width="1118" alt="image" src="https://github.com/user-attachments/assets/bd0a04fc-27f4-4b06-abfc-bc72210d0c08" />

## Feel Free to Contact Me 📞
https://t.me/pa-shk
