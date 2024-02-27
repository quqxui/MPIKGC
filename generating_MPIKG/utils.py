import os
import re
import logging

def log_output(ans):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info('##################################################')
    logging.info(ans)
    logging.info('##################################################\n\n')


def mkdir_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])

def output_refine_llama2(text):
    text = text.split('[/INST]')[1]
    sentences = re.split(r'[.!?]', text)
    new_sentences = []  # 新的句子列表
    for sentence in sentences:
        if "you" in sentence.lower() or 'of course' in sentence.lower():  # "you" 'of course'
            pass
        else:
            new_sentences.append(sentence)
    text = '.'.join(new_sentences)
    return text

def check_str(sentence, filter_list):
    for item in filter_list:
        if item in sentence.lower():
            return True
    return False
