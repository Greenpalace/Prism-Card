#web
# import psycopg2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import argparse
from typing_extensions import Required

#kospeech
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
import os
# from tools import revise
import revise

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

#korquad
import json
from collections import OrderedDict
import re

import random
from argparse import ArgumentParser

import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from korquad_qg.config import QGConfig
from korquad_qg.dataset import MAX_QUESTION_SPACE, MIN_QUESTION_SPACE, QGDecodingDataset, load_korquad_dataset

#kospeech ready

model_path = "C:\Python\kospeech\outputs\2022-08-19\13-44-10\model.pt"
audio_path = ""
device = "cpu"

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'wav') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

def voicedetection() :

# require (x)- > required (o)
    parser = argparse.ArgumentParser(description='KoSpeech')
    parser.add_argument(model_path, type=str, required=True)
    parser.add_argument(audio_path, type=str, required=True)
    parser.add_argument(device, type=str, required=False, default='cpu')
    opt = parser.parse_args()

# 음성 하나에 대해 inference 하는 경우
    feature = parse_audio(opt.audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KsponSpeechVocabulary('cssiri_character_vocabs.csv')

    model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    if isinstance(model, ListenAttendSpell):
        model.encoder.device = opt.device
        model.decoder.device = opt.device

        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, DeepSpeech2):
        model.device = opt.device
        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
        y_hats = model.greedy_search(feature.unsqueeze(0), input_length)

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    return revise(sentence)

#korquad ready

file_data = OrderedDict()
file_data['version'] = 'KorQuAD_v1.0_dev'
title = '(title)'

global output

def questionmake(script):
    outputtexts = []

    context = script
    question = '.'  # 삭제

    context2 = context.replace("\n", "")
    context2 = re.sub(r'[1-5]', '', context2)

    okt = Okt()

    tokenized_doc = okt.pos(context2)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    n_gram_range = (1, 1)

    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([context2])
    candidate_embeddings = model.encode(candidates)

    top_n = 3  # 상위 3개의 키워드를 출력
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    text = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    answer_start = [context2.find(text[0]), context2.find(text[1]), context2.find(text[2])]
    # id가 질문마다 생성되는 거여서 어떻게 부여할지 생각 - 우선 0,1,2으로 해놓음

    file_data['data'] = [{'paragraphs': [{'qas': [
        {'answers': [{'text': text[0], 'answer_start': answer_start[0]}], 'id': 0, 'question': question},
        {'answers': [{'text': text[1], 'answer_start': answer_start[1]}], 'id': 1, 'question': question},
        {'answers': [{'text': text[2], 'answer_start': answer_start[2]}], 'id': 2, 'question': question}],
                                          'context': context2}], 'title': title}]

    data = json.dumps(file_data, ensure_ascii=False)
    data2 = json.loads(data)

    file_path = "data/text.json"

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data2, file)

    MODEL_PATH = "artifacts/gpt2_2022.08.12_13.27.10/gpt2_step_75000.pth"
    OUTPUT_PATH = "decoded.txt"

    parser = ArgumentParser()
    parser.add_argument("-m", MODEL_PATH, type=str, required=True)
    parser.add_argument("-o", OUTPUT_PATH, type=str, required=True)
    parser.add_argument("-s", "--num-samples", type=int)
    parser.add_argument("-b", "--num-beams", type=int, default=5)

    config = QGConfig()
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = SentencePieceBPETokenizer.from_file(
        vocab_filename="tokenizer/vocab.json", merges_filename="tokenizer/merges.txt", add_prefix_space=False
    )
    examples = load_korquad_dataset(config.dev_dataset)
    random.shuffle(examples)
    examples = examples[: args.num_samples]
    dataset = QGDecodingDataset(examples, tokenizer, config.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=1)

    model = model.to(device)
    model.eval()

    generated_results = []

    for i, batch in tqdm(enumerate(dataloader), desc="generate", total=len(dataloader)):
        input_ids, attention_mask = (v.to(device) for v in batch)
        origin_seq_len = input_ids.size(-1)

        decoded_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=origin_seq_len + MAX_QUESTION_SPACE,
            min_length=origin_seq_len + MIN_QUESTION_SPACE,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_beams=args.num_beams,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )

        for decoded_tokens in decoded_sequences.tolist():
            decoded_question_text = tokenizer.decode(decoded_tokens[origin_seq_len:])
            decoded_question_text = decoded_question_text.split("</s>")[0].replace("<s>", "")
            generated_results.append(
                (examples[i].context, examples[i].answer, examples[i].question, decoded_question_text)
            )

    with open(args.OUTPUT_PATH, "w") as f:
        for context, answer, question, generated_question in generated_results:
            outputtexts.append([answer, generated_question])

    return outputtexts


#web ready
app = Flask(__name__)
# connect = psycopg2.connect("dbname=tutorial user=postgres password=1111")
# cur = connect.cursor()  # create cursor

def edit_dist(str1, str2):
    dp = [[0] * (len(str2)+1) for _ in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        dp[i][0] = i
    for j in range(1, len(str2)+1):
        dp[0][j] = j

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

    return dp[-1][-1]  #편집거리 source.

@app.route('/')
def main():
    return render_template("login.html")

@app.route('/iscorrect', methods=['get'])
def iscorrect():

    if edit_dist(voicedetection(), output[0]) < 10 :
        return render_template('correct.html')
    else :
        return render_template('incorrect.html')

    #return render_template('correct.html') #임시로 넣음.

@app.route('/quiz', methods=['get', 'post'])
def quiz():
    global situation
    global script
    situation = request.form["situation"]
    script = request.form["script"]

    output = questionmake(script)

    return render_template("quiz.html", datas = output[1])


@app.route('/modify', methods=['get', 'post'])
def modify():
    global situation
    global script
    situation = request.form["situation"]
    cript = request.form["script"]
    #direction = request.form["direction"]

    '''
    이 부분에서 기존 DB에 있는것을 색인, 삭제함
    '''
    datas=[situation, script]

    return render_template("generate.html", datas=datas)

@app.route('/return_powercard', methods=['get','post'])
def return_powercard():
    #선택한 항목을 그대로 가져옴.
    #datas=[situation, script]

    datas=['예시 문장 1입니다.', '예시 문장 2입니다.'] #예시문장임.

    return render_template("powercard.html", datas=datas)

@app.route('/open_powercard', methods=['get','post'])
def open_powercard():
    global situation
    global script
    situation = request.form["situation"]
    script = request.form["script"]
    #선택한 항목을 그대로 가져옴.
    #datas=[situation, script]

    datas=['예시 문장 1입니다.', '예시 문장 2입니다.'] #예시문장임.

    return render_template("powercard.html", datas=datas)


@app.route('/submit_form', methods=['post'])
def submit_form():
    global situation
    global script
    situation = request.form["situation"]
    script = request.form["script"]
    direction = request.form["direction"]
    '''
    이 부분에서 자연어 처리 AI가 작동함.
    situation, script 변수에 문자열 형태로 입력 데이터 저장,
    이를 모델에 input 하여 나온 output을 DB에 저장함.
    
    이후 DB 색인 기능으로 list를 추출, list.html로 적용함
    list.html에는 예시 항목 2개가 들어 있음. 
    '''

    return render_template("list.html")

@app.route('/file_upload', methods=['get', 'post'])
def file_upload():
    if request.method == 'post':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return '저장되었습니다.'
    else:
        return render_template("generate.html")

@app.route('/select_list', methods=['get', 'post'])
def select_list():
    '''
    이 부분에서 DB 색인으로 만든 파워카드 목록을 추출합니다.
    추출한 데이터를 list.html로 넘겨줍니다.
    '''
    datas = [[1, '제목 예시입니다.', '스크립트 예시입니다. 문장 한 줄로 표시됩니다.'],
             [2, '제목 예시입니다.', '스크립트 예시입니다. 문장 한 줄로 표시됩니다.'],
             [3, '제목 예시입니다.', '스크립트 예시입니다. 문장 한 줄로 표시됩니다.']] #예시입니다.
    return render_template("list.html", datas=datas)


@app.route('/select_generate', methods=['get'])
def select_generate():
    datas = ["상황 설명을 작성해 주세요.", "스크립트를 작성해 주세요."]
    return render_template("generate.html", datas=datas)


@app.route('/login', methods=['post'])
def register():
    global id
    id = request.form["id"]
    password = request.form["password"]
    send = request.form["send"]
    '''
    if send == 'sign up':
        cur.execute("SELECT * FROM users  where id = '{}';".format(id))
        result = cur.fetchall()

        if len(result) == 0:  # not exists
            cur.execute("INSERT INTO users VALUES ('{}', '{}');".format(id, password))
            cur.execute("INSERT INTO account VALUES('{}', 10000, 'beginner');".format(id))

            connect.commit()

            return render_template("main.html")
        else:
            return render_template("ID_collision.html")
    ''' #sign up code
    if send == '로그인':
        '''
        cur.execute("SELECT * from users where id = '{}' and password = '{}';".format(id, password))
        result = cur.fetchall()

        if len(result) == 0:
            return render_template("login_fail.html")
        else:
            cur.execute("SELECT * from account where id = '{}';".format(id))
            ac_result = cur.fetchall()

            cur.execute("SELECT * from items;")
            it_result = cur.fetchall()

            return render_template("login_success.html", accounts=ac_result, items=it_result)
        ''' #DB connection
        return render_template("select.html")

if __name__ == '__main__':
    app.run(port="5000", debug=True)
