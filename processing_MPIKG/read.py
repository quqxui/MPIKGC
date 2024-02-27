import re
import shutil
import os

def check_str(sentence, filter_list):
    for item in filter_list:
        if item in sentence.lower():
            return True
    return False

def mkdir_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def read_ent2ans(refine_path):
    ent2ans = {}
    with open(refine_path+'.txt', 'r') as f:
        for line in f.readlines():
            ent, ans = line.rstrip('\n').split('\t')
            ent2ans[ent] = ans
    return ent2ans


def read_ent2des(datafold, fb_or_wn):
    ent2des = {}
    if fb_or_wn in ['FB13','WN11']:
        filePath = f'{datafold}{fb_or_wn}/entity2text.txt'
    elif fb_or_wn in ['FB15k237']:
        filePath = f'{datafold}{fb_or_wn}/FB15k_mid2description.txt'
    elif fb_or_wn in ['WN18RR']:
        filePath = f'{datafold}{fb_or_wn}/wordnet-mlj12-definitions.txt'

    with open(filePath, 'r') as f:
        for line in f.readlines():
            if fb_or_wn in ['FB15k237']:
                entity, des = line.rstrip('\n').split('\t')
                ent2des[entity] = des[1:].replace('"@en','')
            elif fb_or_wn in ['FB13','WN11']:
                entity, des = line.rstrip('\n').split('\t')
                ent2des[entity] = des
            elif fb_or_wn in ['WN18RR']:
                entity, name, des = line.rstrip('\n').split('\t')
                ent2des[entity] = des

    return ent2des


def read_ent2expa_cot(LLMname, datafold, fb_or_wn, brief_or_expa):
    ent2expa_cot = read_ent2ans(f'{datafold}{fb_or_wn}/{brief_or_expa}')
    new_dict = {}
    for entity, expansion in ent2expa_cot.items():
        if LLMname=='llama2':
            expansion = expansion.split('[/INST]')[1]
            sentences = re.split(r'[:.!?]', expansion)
            new_sentences = []  # 新的句子列表
            for sentence in sentences:
                if not check_str(sentence, ['you','of course','sure','help']):
                    new_sentences.append(sentence)
            expansion = '. '.join(new_sentences)
        elif LLMname=='chatglm2':
            pass
        if len(expansion) ==0: 
            expansion = '[UNK]'
        new_dict[entity] = expansion
    return new_dict

def read_ent2keyword(LLMname, datafold, fb_or_wn, src_or_expa):
    ent2keyword = read_ent2ans(f'{datafold}{fb_or_wn}/{src_or_expa}')
    new_dict = {}
    for entity, keyword in ent2keyword.items():
        if LLMname=='llama2':
            keyword = keyword.split('[/INST]')[1]
            if ':' in keyword:
                keyword = keyword.split(':')[1]
        elif LLMname=='chatglm2':
            pass
        keyword = re.sub(r'[\d\.]+', '', keyword)
        if len(keyword) ==0: 
            keyword = '[UNK]'
        new_dict[entity] = keyword
    return new_dict


def read_rel2des(LLMname, datafold, fb_or_wn, rel_version):
    rel2des = {}
    with open(f'{datafold}{fb_or_wn}/{rel_version}.txt', 'r') as f:
        for line in f.readlines():
            rel,des = line.strip().split('\t')
            if LLMname=='llama2':
                if fb_or_wn not in ['WN18RR']: # manual
                    des = des.split('[/INST]')[1]
                sentences = re.split(r'[:.!?]', des)
                new_sentences = []  # 新的句子列表
                for sentence in sentences:
                    if not check_str(sentence, ['you','of course','sure','help']):
                        new_sentences.append(sentence)
                des = '. '.join(new_sentences)
            elif LLMname=='chatglm2':
                pass
            if fb_or_wn in ['FB13','WN11','WN18RR']:
                des = des.replace(' ','_')
                rel2des[rel] = rel+ '_:_' + des
            elif fb_or_wn in ['FB15k237']:
                des = des.replace(' ','/')
                rel2des[rel] = rel+ '/:/' + des
    return rel2des

def data_copy(datafold, dataset, fb_or_wn):
    mkdir_path(datafold + dataset)
    if fb_or_wn == 'FB13':
        src_files = ['train.tsv','test.tsv','dev.tsv','entity2text.txt', 'entity2textshort.txt','entity2text_capital.txt','relations.txt','entities.txt','relation2text.txt',]
    elif fb_or_wn == 'WN11':
        src_files = ['train.tsv','test.tsv','dev.tsv','entity2text.txt','relations.txt','entities.txt','relation2text.txt',]
    elif fb_or_wn == 'FB15k237':
        src_files = ['train.txt','test.txt','valid.txt','FB15k_mid2name.txt','FB15k_mid2description.txt']
    elif fb_or_wn == 'WN18RR':
        src_files = ['train.txt','test.txt','valid.txt','wordnet-mlj12-definitions.txt']
    for src_file in src_files:
        src_path = f'{datafold}{fb_or_wn}/{src_file}'
        shutil.copy(src_path, datafold + dataset)

def change_triplefile_relation(LLMname, datafold, dataset, fb_or_wn, rel_version):
    rel2new_des_list = []
    for rel_ver in rel_version:
        out_dict = read_rel2des(LLMname,datafold, fb_or_wn, rel_ver)
        rel2new_des_list.append(out_dict)

    rel2new_des_all_list = {}
    for rel in rel2new_des_list[0].keys():
        values = []
        for rel2new_des in rel2new_des_list:
            values.append(rel2new_des[rel])
        if fb_or_wn in ['FB13','WN11','WN18RR']:
            values = '_[SEP]_'.join(values)
        elif fb_or_wn in ['FB15k237']:
            values = '/[SEP]/'.join(values)
        rel2new_des_all_list[rel] = values

    if fb_or_wn in ['FB13','WN11']:
        suffix = '.tsv'
        dev_or_valid = 'dev'
    if fb_or_wn in ['FB15k237','WN18RR']:
        suffix = '.txt'
        dev_or_valid = 'valid'
    for File in [f'train{suffix}',f'test{suffix}',f'{dev_or_valid}{suffix}']:
        with open(datafold + dataset + File, 'w') as file_write:
            with open(f'{datafold}{fb_or_wn}/{File}','r') as file:
                lines = file.readlines()
                for i in range(len(lines)):
                        if fb_or_wn in ['FB13','WN11'] and File in [f'test{suffix}',f'dev{suffix}']:
                            h,rel, t, label = lines[i].strip('\n').split('\t')
                            lines[i] = f"{h}\t{rel2new_des_all_list[rel]}\t{t}\t{label}\n"
                        else:
                            h,rel, t = lines[i].strip('\n').split('\t')
                            lines[i] = f"{h}\t{rel2new_des_all_list[rel]}\t{t}\n"
                file_write.writelines(lines)

