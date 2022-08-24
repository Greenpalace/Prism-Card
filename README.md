
# DATA 09

데이터청년캠퍼스 9조의 서비스입니다! 저희는 자폐 아동을 위한 교육 검증 서비스를 목표로 해당 서비스를 디자인했습니다.
이를 위해 kospeech, KoNLPy, KeyBERT를 기반으로 작업하였습니다.

kospeech : [한국어 기반 STT 모델](https://github.com/sooftware/kospeech/)

KoNLPy : [한국어 정보처리를 위한 NLP 모델](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)

KeyBERT : [자연어 키워드 추출 모델](https://github.com/MaartenGr/KeyBERT)

해당 패키지를 통한 학습을 위해 다음과 같은 데이터를 사용하였습니다.

KorQuAD : [한국어 데이터셋](https://korquad.github.io/KorQuad%201.0/)

AIhub : [한국어 아동 음성 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=540)

# 설치

해당 웹서비스의 구동을 위해 다음과 같은 패키지가 사용되었습니다.

* Numpy: `pip install numpy` (Numpy 설치에 문제가 생긴다면 [이곳](https://github.com/numpy/numpy)을 참조하세요).
* Pytorch: `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch` (설치에 문제가 생긴다면 [이곳](http://pytorch.org/)을 참조하세요.)
* Pandas: `pip install pandas` (Pandas 설치에 문제가 생긴다면 [이곳](https://github.com/pandas-dev/pandas)을 참조하세요.)  
* Matplotlib: `pip install matplotlib` (Matplotlib 설치에 문제가 생긴다면 [이곳](https://github.com/matplotlib/matplotlib)을 참조하세요)
* librosa: `conda install -c conda-forge librosa` (librosa 설치에 문제가 생긴다면 [이곳](https://github.com/librosa/librosa)을 참조하세요 )
* tqdm: `pip install tqdm` (tqdm 설치에 문제가 생긴다면 [이곳](https://github.com/tqdm/tqdm)을 참조하세요)
* sentencepiece: `pip install sentencepiece` (sentencepiece 설치에 문제가 생긴다면 [이곳](https://github.com/google/sentencepiece)을 참조하세요)
* hydra: `pip install hydra-core==1.1.1` (hydra 설치에 문제가 생긴다면 [이곳](https://github.com/facebookresearch/hydra)을 참조하세요)
