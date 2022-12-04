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
