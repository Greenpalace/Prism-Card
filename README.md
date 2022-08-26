
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
* transformers: `pip install transformers` (transformers 설치에 문제가 생긴다면 [이곳](https://github.com/huggingface/transformers)을 참조하세요)
* tokenizers: `pip install tokenizers` (tokenizers 설치에 문제가 생긴다면 [이곳](https://github.com/huggingface/tokenizers)을 참조하세요)
* KoNLPy : `pip install konlpy` (KoNLPy 설치에 문제가 생긴다면 [이곳](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)을 참조하세요)

# kospeech, KorQuAD 설치 방법

해당 패키지의 설치를 위해 kospeech, korquad 파일을 경로로 지정하고 

```
pip install -e.
```
을 통해 패키지를 설치하시면 됩니다. 


# 함수 설명

각 필요 함수들은 main() 이전 * ready로 필요한 부분들을 정리해 두었습니다.

- voicedetection()  : 음성 인식이 작동되는 함수입니다. (from kospeech)
- questionmake()    : 질문 생성이 작동되는 함수입니다. (from korquad)

각 모듈들은 Flask 를 통해 내부 계산 후 html web에 출력됩니다.

- edit_dist()       : 편집거리 함수입니다. 질문 페이지에서 정답/오답을 판단할때 이용됩니다.

# 구조 설명

전체적 구조는 Flask를 이용해 app.py가 내부 계산을 수행하고, render_template()를 통해 html이 작동됩니다. Flask는 .py와 html 간 데이터 교환을 가능하게 합니다.

각 html 메소드는 @app.route()와 def를 통하여 지정되어 html에서 실행됩니다.

- 메소드 설명
@app.route('\') : 기본 실행입니다. login.html을 불러옵니다.
@app.route('/login') : 로그인 기능을 담당합니다. DB와 연동하여 로그인하고 select.html을 불러옵니다.
@app.route('/select_generate') : select.html에서 생성을 선택할 경우 실행됩니다. generate.html을 불러옵니다.
@app.route('/select_list') : select.html에서 목록을 선택할 경우 실행됩니다. DB에서 파워카드 목록을 추출하여 이 데이터를 list.html로 넘겨줍니다. list.html을 불러옵니다.
@app.route('/file_upload') : generate.html에서 파일 업로드를 담당하는 메소드입니다.
@app.route('/submit_form') : generate.html에서 선택,작성한 정보를 DB에 저장하고 목록을 추출해 list.html을 불러옵니다.
@app.route('/open_powercard') : 파워카드를 생성합니다. powercard.html을 불러옵니다.
@app.route('/return_powercard') : powercard.html을 불러옵니다.
@app.route('/modify') : powercard.html에서 파워카드를 수정할 때 실행됩니다. 기존 정보를 generate.html에서 보여주고, 수정한 정보는 DB에서 수정됩니다.
@app.route('/quiz') : questionmake() 함수를 실행하고 질문, 답을 생성합니다. quiz.html을 불러옵니다.
@app.route('/iscorrect') : 편집거리를 이용해 정답 분기점을 만들어 맞을 경우 correct.html, 틀릴 경우 incorrect.html을 불러옵니다.

각 html은 templates 폴더에, 이미지와 css는 static 폴더에 저장되어 있습니다. javascript code의 경우 html에 포함되어 있습니다.


