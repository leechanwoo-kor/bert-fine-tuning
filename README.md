# bert-fine-tuning

# 목차

- [목차](#목차)

- [서론](#서론)
    - [BERT란 무엇인가?](#bert란-무엇인가)
    - [Fine-Tuning의 장점](#fine-tuning의-장점)
        - [NLP에서의 변화](#nlp에서의-변화)
- [1. 설정](#1-설정)
    - [1.1. Colab에서 학습을 위한 GPU 사용](#11-colab에서-학습을-위한-gpu-사용)
    - [1.2. Hugging Face 라이브러리 설치](#12-hugging-face-라이브러리-설치)
- [2. CoLA 데이터셋 불러오기](#2-cola-데이터셋-불러오기)
    - [2.1. 다운로드 및 추출](#21-다운로드-및-추출)
    - [2.2. 구문 분석](#22-구문-분석)
- [3. 토큰화 및 입력 포맷](#3-토큰화-및-입력-포맷)
    - [3.1. BERT 토큰화기](#31-bert-토큰화기)
    - [3.2. Required Formatting](#32-required-formatting)
        - [스페셜 토큰](#스페셜-토큰)
        - [문장 길이 및 어텐션 마스크](#문장-길이-및-어텐션-마스크)
    - [3.3. 데이터 세트 토큰화](#33-데이터-세트-토큰화)
    - [3.4. 학습 및 검증 분할](#34-학습-및-검증-분할)

# 서론

2018년은 NLP에서 획기적인 해였다. Allen AI의 ELMO, OpenAI의 Open-GPT와 구글의 BERT와 같은 모델은 연구자들이 최소한의 fine-tuning으로 기존 벤치마크하던 모델을 능가했다. 그리고 더 적은 데이터와 더 적은 계산 시간으로 pre-training된 모델을 제공하여 쉽게 fine-tuning된 우수한 성능을 생성할 수 있었다.

하지만 NLP를 시작한 많은 사람들과 심지어 일부 숙련된 실무자들도 이러한 강력한 모델의 이론과 실제 적용은 잘 이해되지 않는다.

## BERT란 무엇인가?

2018년 말에 출시된 BERT(Bidirectional Encoder Representations from Transformers)는 NLP에서 transfer learning 모델을 사용하기 위한 더 나은 이해와 실용적인 지침을 제공하기 위해 사용할 모델이다. BERT는 언어 표현을 pre training하는 방법으로 모델을 만드는 데 사용되었다. 이러한 모델을 통해 텍스트 데이터에서 특징을 추출하거나 분류, 질의응답 등에 사용할 수 있다.

## Fine-Tuning의 장점

BERT를 사용하여 텍스트 분류기를 학습합니다. 구체적으로 pre-training된 BERT 모델의 끝에 학습되지 않은 뉴런 층을 추가하고 분류 작업을 위한 새로운 모델을 훈련시킬 것입니다. 이것이 특정 NLP 작업에 특화된 CNN, BiLSTM 등과 같은 딥러닝 모델을 훈련 시키는 것보다 좋은 이유는 다음과 같다.

1. 빠른 개발
    - 첫째, pre-training된 BERT 모델 가중치는 이미 우리 언어에 대한 많은 정보를 인코딩한다. 결과적으로 fine-tuning된 모델을 훈련하는 데 훨씬 적은 시간이 소요된다. 이는 이미 네트워크의 하단 계층을 광범위하게 훈련한 것과 같으며 분류 작업에 대한 기능으로 출력을 사용하면서 조정만 하면 된다.

2. 적은 데이터
    - 또한 pre-training된 가중치 때문에 이 방법을 사용하면 처음부터 구축된 모델에 필요한 것보다 훨씬 작은 데이터 세트에서 작업을 fine-tuning할 수 있다. 처음부터 구축된 NLP 모델의 주요 단점은 네트워크를 합리적인 정확도로 훈련시키기 위해 종종 엄청나게 큰 데이터 세트가 필요하다는 것이다. 즉, 데이터 세트를 만드는 데 많은 시간과 에너지가 투입되어야 한다는 것이다. BERT를 fine-tuning함으로써 이제 훨씬 적은 양의 학습 데이터에서 우수한 성능을 발휘하도록 모델을 교육하는 것에서 벗어날 수 있다.

3. 더 좋은 결과
    - 마지막으로, 이 간단한 fine-tuning 절차는 분류, 언어 추론, 의미론적 유사성, 질의 응답 등 다양한 작업에 대한 최소한의 작업별 조정으로 우수한 결과를 달성하는 것으로 나타났다. 특정 작업에서 잘 작동하는 것으로 표시된 사용자 지정 및 때로는 모호한 아키텍처를 구현하기보다는 단순히 BERT를 fine-tuning하는 것이 더 나은 또는 최소한 동일한 대안인 것으로 나타났다.

### NLP에서의 변화

transfer learning으로의 이러한 변화는 이전에 컴퓨터 비전에서 일어난 것과 같은 변화와 유사하다. 컴퓨터 비전 작업을 위한 좋은 딥 러닝 네트워크를 만드는 것은 수백만 개의 매개 변수를 필요로 하며 훈련하는 데 매우 비용이 많이 든다.

연구자들은 심층 네트워크가 낮은 계층은 단순한 특징을 높은 계층은 점점 더 복잡한 특징을 갖는 계층적 특징 표현을 학습한다는 것을 발견했다. 매번 새로운 네트워크를 처음부터 훈련시키는 대신, 일반화된 이미지 기능을 가진 훈련된 네트워크의 하위 계층을 복사하고 전송하여 다른 작업을 수행하는 다른 네트워크에서 사용할 수 있다.

pre-training된 심층 네트워크를 다운로드하고 새로운 작업을 위해 신속하게 재학습하거나 네트워크를 처음부터 훈련시키는 값비싼 프로세스보다 훨씬 더 나은 추가 계층을 추가하는 것이 곧 일반적인 관행이 되었다.

2018년 ELMO, BERT, ULMFIT, Open-GPT와 같은 deep pre-trained language model의 도입은 컴퓨터 비전에서 일어난 것과 처럼 NLP에서 transfer learning것과 동일한 변화를 나타낸다.


# 1. 설정

## 1.1. Colab에서 학습을 위한 GPU 사용

Google Colab은 GPU와 TPU를 무료로 제공합니다. 대규모 신경망을 훈련할 것이기 때문에 이를 활용하는 것이 최선이다.(이 경우 GPU를 연결할 것이다), 그렇지 않으면 훈련이 매우 오래 걸릴 것이다.

GPU는 메뉴로 이동하여 다음을 선택하여 추가할 수 있습니다:

`Edit 🡒 Notebook Settings 🡒 Hardware accelerator 🡒 (GPU)`

그런 다음 다음 셀을 실행하여 GPU가 인식되는지 확인합니다.

```Python
import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
```

```
Found GPU at: /device:GPU:0
```

torch가 GPU를 사용하기 위해서는 GPU를 장치로 식별하고 지정해야 한다. 나중에, 우리의 훈련 루프에서, 우리는 데이터를 장치에 로드할 것이다.

```Python
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```

```
There are 1 GPU(s) available.
We will use the GPU: Tesla T4
```

## 1.2. Hugging Face 라이브러리 설치

다음으로, BERT와 함께 작업하기 위한 pytorch 인터페이스를 제공하는 Hugging Face의 [transformers](https://github.com/huggingface/transformers) 패키지를 설치합시다. (이 라이브러리에는 Open AI의 GPT 및 GPT-2와 같은 다른 pre-training된 언어 모델을 위한 인터페이스가 포함되어 있습니다.)

우리가 pytorch 인터페이스를 선택한 이유는 높은 수준의 API(사용하기 쉽지만 어떻게 작동하는지에 대한 통찰력을 제공하지 않음)와 tensorflow 코드때문이다.

현재, Hugging Face 라이브러리는 BERT와 함께 작업하기 위한 가장 널리 받아들여지고 강력한 pytorch 인터페이스인 것 같다. 라이브러리는 다양한 pre-training된 transformers 모델을 지원할 뿐만 아니라 특정 작업에 적합한 이러한 모델의 사전 구축된 수정도 포함한다. 예를 들어 이번 분석에서는 `BertForSequenceClassification`을 사용합니다.

또한 라이브러리에는 토큰 분류, 질문 답변, 다음 문장 예측 등을 위한 작업별 클래스가 포함되어 있다. 이러한 미리 작성된 클래스를 사용하면 목적에 맞게 BERT를 수정하는 프로세스가 간소화됩니다.

```Python
!pip install transformers
```

```
[간소화를 위해 이 output은 삭제했습니다.]
```


# 2. CoLA 데이터셋 불러오기

단일 문장 분류를 위해 The Corpus of Linguistic Acceptability(CoLA) 데이터 세트를 사용합니다. 문법적으로 정확하거나 잘못된 것으로 레이블이 지정된 일련의 문장입니다. 2018년 5월에 처음 공개되었으며 BERT와 같은 모델이 경쟁하는 "GLUE Benchmark"에 포함된 테스트 중 하나입니다.

## 2.1. 다운로드 및 추출

wget 패키지를 사용하여 Colab 인스턴스의 파일 시스템에 데이터세트를 다운로드합니다.

```Python
!pip install wget
```

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting wget
  Downloading wget-3.2.zip (10 kB)
Building wheels for collected packages: wget
  Building wheel for wget (setup.py) ... done
  Created wheel for wget: filename=wget-3.2-py3-none-any.whl size=9674 sha256=6fcd1ffa5fc23b7fc7a11fa4963cacd3d4675bf30315606398577e4ae13b318a
  Stored in directory: /root/.cache/pip/wheels/bd/a8/c3/3cf2c14a1837a4e04bd98631724e81f33f462d86a1d895fae0
Successfully built wget
Installing collected packages: wget
Successfully installed wget-3.2
```

데이터베이스는 깃허브의 [https://nyu-mll.github.io/CoLA/](https://nyu-mll.github.io/CoLA/)에서 호스팅됩니다.

```Python
import wget
import os

print('Downloading dataset...')

# The URL for the dataset zip file.
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# Download the file (if we haven't already)
if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
```

```
Downloading dataset...
```

파일 시스템에 데이터 세트의 압축을 풉니다. 왼쪽 사이드바에서 Colab 인스턴스의 파일 시스템을 탐색할 수 있습니다.

```Python
# Unzip the dataset (if we haven't already)
if not os.path.exists('./cola_public/'):
    !unzip cola_public_1.1.zip
```

```
Archive:  cola_public_1.1.zip
   creating: cola_public/
  inflating: cola_public/README      
   creating: cola_public/tokenized/
  inflating: cola_public/tokenized/in_domain_dev.tsv  
  inflating: cola_public/tokenized/in_domain_train.tsv  
  inflating: cola_public/tokenized/out_of_domain_dev.tsv  
   creating: cola_public/raw/
  inflating: cola_public/raw/in_domain_dev.tsv  
  inflating: cola_public/raw/in_domain_train.tsv  
  inflating: cola_public/raw/out_of_domain_dev.tsv  
```

## 2.2. 구문 분석

파일 이름을 보면 `tokenized` 버전과 `raw` 버전의 데이터를 모두 사용할 수 있음을 알 수 있습니다.

pre-training된 BERT를 적용하기 위해서는 모델이 제공하는 토크나이저를 사용해야 하기 때문에 사전 토큰화된 버전을 사용할 수 없다. 이는 (1) 모델이 특정하고 고정된 어휘를 가지고 있고 (2) BERT 토크나이저가 OOV(out-of-vocabulary)를 처리하는 특별한 방법을 가지고 있기 때문이다.

```Python
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(10)
```

```
Number of training sentences: 8,551
```

||sentence_source|label|label_notes|sentence|
|:---:|:---:|:---:|:---:|:---:|
|8200|ad03|1|NaN|They kicked themselves|
|3862|ks08|1|NaN|A big green insect flew into the soup.|
|8298|ad03|1|NaN|I often have a cold.|
|6542|g_81|0|*|Which did you buy the table supported the book?|
|722|bc01|0|*|Home was gone by John.|
|3693|ks08|1|NaN|I think that person we met last week is insane.|
|6283|c_13|1|NaN|Kathleen really hates her job.|
|4118|ks08|1|NaN|Do not use these words in the beginning of a s...|
|2592|l-93|1|NaN|Jessica sprayed paint under the table.|
|8194|ad03|0|*|I sent she away.|

우리가 실제로 관심을 갖는 두 가지 속성은 `문장`과 그 `라벨`이며, 이를 "수용성 판단(acceptability judgment)"(0=불수용(unacceptable), 1=수용(acceptable))이라고 한다.

여기 문법적으로 허용되지 않는 것으로 분류된 다섯 개의 문장이 있다. 감정 분석과 같은 것보다 이 작업이 얼마나 더 어려운지 볼 수 있습니다.

```Python
df.loc[df.label == 0].sample(5)[['sentence', 'label']]
```

||sentence|label|
|:---:|:---:|:---:|
|4867|They investigated.|0|
|200|The more he reads, the more books I wonder to ...|0|
|4593|Any zebras can't fly.|0|
|3226|Cities destroy easily.|0|
|7337|The time elapsed the day.|0|

학습 세트의 문장과 레이블을 숫자 배열로 추출해 봅시다.

```Python
# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values
```


# 3. 토큰화 및 입력 포맷

이 섹션에서는 데이터 세트를 BERT가 훈련할 수 있는 형식으로 변환할 것이다.

## 3.1. BERT 토큰화기

BERT에 텍스트를 공급하려면 토큰으로 분할한 다음 토큰을 토큰화기 어휘의 인덱스에 매핑해야 합니다.

토큰화는 BERT에 포함된 토큰화기에서 수행해야 합니다. 아래 셀에서 다운로드합니다. 여기서는 "케이스 없는" 버전을 사용할 것입니다.

```Python
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```

```
Loading BERT tokenizer...
Downloading: 100%
232k/232k [00:00<00:00, 3.15MB/s]
Downloading: 100%
28.0/28.0 [00:00<00:00, 380B/s]
Downloading: 100%
570/570 [00:00<00:00, 7.98kB/s]
```

출력을 보기 위해 한 문장에 토큰화기를 적용해 봅시다.

```Python
# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
```

```
 Original:  Our friends won't buy this analysis, let alone the next one we propose.
Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]
```

실제로 모든 문장을 변환할 때 `tokenize`와 `convert_tokens_to_ids`를 별도로 호출하는 대신 `tokenize.encode` 함수를 사용하여 두 단계를 모두 처리합니다.

그러나 이를 수행하기 전에 BERT의 포맷 요구 사항 중 일부에 대해 설명해야 합니다.

## 3.2. Required Formatting

위의 코드에서는 여기서 살펴볼 몇 가지 필수 형식 지정 단계를 생략했습니다.

*참고 사항: BERT에 대한 입력 형식은 내가 보기에 "과잉 지정"된 것처럼 보입니다. 중복되는 것처럼 보이거나 데이터에서 쉽게 추론할 수 있는 여러 정보를 제공해야 합니다. 우리가 명시적으로 제공하지 않아도 됩니다. 하지만 그것은 사실이고, 나는 내가 BERT 내부를 더 깊이 이해하게 되면 더 말이 될 것이라고 생각한다.*

다음을 수행해야 합니다.

1. 각 문장의 시작과 끝에 특별한 토큰을 추가하세요.
2. 모든 문장을 하나의 일정한 길이로 패드하고 자릅니다.
3. "attention mask"를 사용하여 실제 토큰과 패딩 토큰을 명시적으로 구별합니다.

### 스페셜 토큰

`[SEP]`

모든 문장 끝에 스페셜 `[SEP]` 토큰을 추가해야 합니다.

이 토큰은 BERT에 두 개의 개별 문장이 주어지고 무언가를 결정하도록 요청되는 두 문장 작업의 아티팩트이다(예: 문장 A의 질문에 대한 답을 문장 B에서 찾을 수 있습니까?).

우리가 단문 입력만 있는데 토큰이 왜 아직도 필요한지는 아직 확실하지 않지만, 그렇습니다!

<br>

`[CLS]`

분류 작업의 경우 모든 문장의 시작 부분에 스페셜 `[CLS]` 토큰을 추가해야 한다.

이 토큰은 특별한 의미가 있습니다. BERT는 12개의 트랜스포머 레이어로 구성됩니다. 각 트랜스포머는 토큰 임베딩 목록을 가져와 출력에 동일한 수의 임베딩을 생성한다(물론 피쳐 값이 변경된다!).

![image](https://user-images.githubusercontent.com/55765292/205487650-7876f63a-e42a-48c6-aede-15939b00a059.png)

최종(12번째) 트랜스포머의 출력에서 분류기는 첫 번째 임베딩([CLS] 토큰에 해당)만 사용한다.

> "모든 시퀀스의 첫 번째 토큰은 항상 스페셜 분류 토큰([CLS])입니다. 이 토큰에 해당하는 최종 은닉 상태는 분류 작업의 집계 시퀀스 표현으로 사용됩니다."([BERT 논문](https://arxiv.org/pdf/1810.04805.pdf)에서)

최종 임베딩에 대해 풀링 전략을 시도해 볼 수 있지만, 이것은 필요하지 않다. BERT는 분류에만 이 [CLS] 토큰을 사용하도록 훈련되었기 때문에, 우리는 모델이 분류 단계에 필요한 모든 것을 단일 768 값 임베딩 벡터로 인코딩하도록 동기를 부여했다는 것을 알고 있다. 우리를 위한 풀링은 이미 끝났어요!

### 문장 길이 및 어텐션 마스크

데이터 세트의 문장은 분명히 다양한 길이를 가지고 있는데, BERT는 이것을 어떻게 처리할까요?

BERT에는 두 가지 제약 조건이 있습니다.
- 모든 문장은 고정된 단일 길이로 패딩되거나 잘려야 한다.
- 최대 문장 길이는 512 토큰입니다.

패딩은 BERT 어휘에서 인덱스 0에 있는 스페셜 `[PAD]` 토큰으로 수행됩니다. 아래 그림은 8개의 토큰 "MAX_LEN"으로 패딩하는 방법을 보여줍니다.

![image](https://user-images.githubusercontent.com/55765292/205487988-f45c3a9f-fbb0-4197-a027-2dbda1f27102.png)

"어텐션 마스크"는 단순히 어떤 토큰이 패딩이고 어떤 토큰이 패딩이 아닌지를 나타내는 1과 0의 배열입니다. (약간 중복되는 것 같지 않나요?) 이 마스크는 BERT의 "Self-Attention" 메커니즘에 이러한 PAD 토큰을 문장 해석에 통합하지 말라고 말한다.

그러나 최대 길이는 훈련과 평가 속도에 영향을 미친다. 예를 들어, Tesla K80의 경우:

`MAX_LEN = 128 --> Training epochs take ~5:28 each`

`MAX_LEN = 64 --> Training epochs take ~2:57 each`


## 3.3. 데이터 세트 토큰화

트랜스포머 라이브러리는 대부분의 구문 분석 및 데이터 준비 단계를 처리할 수 있는 유용한 `encode` 함수를 제공합니다.

그러나 텍스트를 인코딩할 준비가 되기 전에 패딩/잘라내기를 위한 **최대 문장 길이**를 결정해야 합니다.

아래 셀은 최대 문장 길이를 측정하기 위해 데이터 세트의 토큰화 패스를 하나 수행합니다.

```Python
max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
```

```
Max sentence length:  47
```

혹시 더 긴 시험 문장이 있을 경우를 대비해서 최대 길이를 64로 설정하겠습니다.

이제 실제 토큰화를 수행할 준비가 되었습니다.

`tokenizer.encode_plus` 함수는 다음과 같은 여러 단계를 결합합니다.

- 문장을 토큰으로 분할합니다.
- 스페셜 `[CLS]` 및 `[SEP]` 토큰을 추가합니다.
- 토큰을 ID에 매핑합니다.
- 모든 문장을 같은 길이로 패드하거나 자릅니다.
- 실제 토큰을 `[PAD]` 토큰과 명시적으로 구별하는 어텐션 마스크를 만듭니다.

처음 네 가지 기능은 `tokenizer.encode`에 있지만 다섯 번째 항목(어텐션 마스크)을 얻기 위해 `tokenizer.encode_plus`를 사용하고 있습니다. 문서는 [여기](https://huggingface.co/docs/transformers/main_classes/tokenizer?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus)에 있습니다.

```Python
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])
```

```
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
/usr/local/lib/python3.8/dist-packages/transformers/tokenization_utils_base.py:2336: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
  warnings.warn(
Original:  Our friends won't buy this analysis, let alone the next one we propose.
Token IDs: tensor([  101,  2256,  2814,  2180,  1005,  1056,  4965,  2023,  4106,  1010,
         2292,  2894,  1996,  2279,  2028,  2057, 16599,  1012,   102,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0])
```


## 3.4. 학습 및 검증 분할
학습에 90%, 검증에 10%를 사용하도록 학습 세트를 나눕니다.

```Python
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
```

```
7,695 training samples
  856 validation samples
```

우리는 또한 torch DataLoader 클래스를 사용하여 데이터 세트에 대한 iterator를 만들 것이다. 이것은 for 루프와 달리 iterator를 사용하면 전체 데이터 세트를 메모리에 로드할 필요가 없기 때문에 훈련 중 메모리를 절약하는 데 도움이 된다.

```Python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
```

