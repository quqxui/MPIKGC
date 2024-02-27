import os
import shutil


def CSprom_fb(datasetname):
    filepath = './../CSProm-KG/data/raw/'
    if os.path.exists(filepath + datasetname):
        shutil.rmtree(filepath + datasetname)
    shutil.copytree(LLMfold + datasetname, filepath + datasetname)
    rename_dict = {'train.txt':'train.tsv',
                   'test.txt':'test.tsv',
                   'valid.txt':'dev.tsv',
                   'FB15k_mid2description.txt':'entity2textlong.txt',
                   'FB15k_mid2name.txt':'entity2text.txt'}
    for src, tar in rename_dict.items():
        if os.path.exists(filepath + datasetname + src):
            os.rename(filepath + datasetname + src, filepath + datasetname + tar)

    # replace _ in etity2text 
    etity2text = {}
    with open(filepath + datasetname + 'entity2text.txt', 'r') as file:
        for line in  file.readlines():
            entity, text = line.strip('\n').split('\t')
            etity2text[entity] = text.replace('_', ' ')
    with open(filepath + datasetname + 'entity2text.txt', 'w') as file:
        for entity, text in etity2text.items():
            file.write(f"{entity}\t{text}\n")

    entities = set()
    relation2text = {}
    for file in ['train.tsv','test.tsv','dev.tsv']:
        with open(filepath + datasetname +file, 'r') as f:
            for line in f.readlines():
                h,r,t  = line.strip('\n').split('\t')
                entities.add(h)
                entities.add(t)
                relation2text[r] = r.replace('/',' ')

    with open(filepath + datasetname + 'relation2text.txt', 'w') as f:
        for r,text in relation2text.items():
            f.write(f"{r}\t{text}\n")
    with open(filepath + datasetname + 'entities.txt', 'w') as f:
        for ent in entities:
            f.write(f"{ent}\n")


def CSprom_wn(datasetname):

    filepath = './../CSProm-KG/data/raw/'
    if os.path.exists(filepath + datasetname):
        shutil.rmtree(filepath + datasetname)
    shutil.copytree(LLMfold + datasetname, filepath + datasetname)
    rename_dict = {'train.txt':'train.tsv',
                   'test.txt':'test.tsv',
                   'valid.txt':'dev.tsv',
                   }
    for src, tar in rename_dict.items():
        if os.path.exists(filepath + datasetname + src):
            os.rename(filepath + datasetname + src, filepath + datasetname + tar)

    entities = set()
    relations = set()
    for file in ['train.tsv','test.tsv','dev.tsv']:
        with open(filepath + datasetname +file, 'r') as f:
            for line in f.readlines():
                h,r,t  = line.strip('\n').split('\t')
                entities.add(h)
                entities.add(t)
                relations.add(r)

    with open(filepath + datasetname + 'relations.txt', 'w') as f:
        for r in relations:
            f.write(f"{r}\n")
    with open(filepath + datasetname + 'entities.txt', 'w') as f:
        for ent in entities:
            f.write(f"{ent}\n")


def LMKE_wn(datasetname):
    filepath = './../LMKE/data/'
    if os.path.exists(filepath + datasetname):
        shutil.rmtree(filepath + datasetname)
    shutil.copytree(LLMfold + datasetname, filepath + datasetname)
    rename_dict = {'train.txt':'train.tsv',
                   'test.txt':'test.tsv',
                   'valid.txt':'dev.tsv',
                   }
    for src, tar in rename_dict.items():
        if os.path.exists(filepath + datasetname + src):
            os.rename(filepath + datasetname + src, filepath + datasetname + tar)

    # relation2text.txt
    relation2text = {}
    for file in ['train.tsv','test.tsv','dev.tsv']:
        with open(filepath + datasetname +file, 'r') as f:
            for line in f.readlines():
                h,r,t  = line.strip('\n').split('\t')
                relation2text[r] = ' '.join(r.split('_'))
    with open(filepath + datasetname + 'relation2text.txt', 'w') as f:
        for r,text in relation2text.items():
            f.write(f"{r}\t{text}\n")

    # my_entity2text.txt
    in_path = filepath + datasetname + 'wordnet-mlj12-definitions.txt'
    out_path = filepath + datasetname +  'my_entity2text.txt'
    descriptions = {}
    with open(in_path, 'r', encoding='utf8') as fil:
        for line in fil.readlines():
            idx, name, description = line.strip('\n').split('\t')
            name = ' '.join([ s for s in name.split('_')[:-2] if s != ''])
            descriptions[idx] = name +' : ' + description 
    with open(out_path, 'w', encoding='utf8') as fil:
        for k, v in descriptions.items():
            fil.write('{}\t{}\n'.format(k, v))

    # delete wordnet-mlj12-definitions.txt
    os.remove(in_path)

def LMKE_fb(datasetname):
    filepath = './../LMKE/data/'
    if os.path.exists(filepath + datasetname):
        shutil.rmtree(filepath + datasetname)
    shutil.copytree(LLMfold + datasetname, filepath + datasetname)
    rename_dict = {'train.txt':'train.tsv',
                   'test.txt':'test.tsv',
                   'valid.txt':'dev.tsv',
                   }
    for src, tar in rename_dict.items():
        if os.path.exists(filepath + datasetname + src):
            os.rename(filepath + datasetname + src, filepath + datasetname + tar)

    relation2text = {}
    for file in ['train.tsv','test.tsv','dev.tsv']:
        with open(filepath + datasetname +file, 'r') as f:
            for line in f.readlines():
                h,r,t  = line.strip('\n').split('\t')
                relation2text[r] = ' '.join(r.split('/'))
    with open(filepath + datasetname + 'relation2text.txt', 'w') as f:
        for r,text in relation2text.items():
            f.write(f"{r}\t{text}\n")


def SimKGC():
    src_folder = LLMfold
    dst_folder = './../SimKGC/data'
    for folder_name in os.listdir(src_folder):
        if folder_name.startswith('FB15k237_') or folder_name.startswith('WN18RR_'):
            src_path = os.path.join(src_folder, folder_name)
            dst_path = os.path.join(dst_folder, folder_name)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


def TC():
    src_folder = LLMfold
    dst_folder = './../LMKE/data'
    for folder_name in os.listdir(src_folder):
        if folder_name.startswith('FB13_') or folder_name.startswith('WN11_'):
            src_path = os.path.join(src_folder, folder_name)
            dst_path = os.path.join(dst_folder, folder_name)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)

    dst_folder = './../kg-bert/data'
    for folder_name in os.listdir(src_folder):
        if folder_name.startswith('FB13_') or folder_name.startswith('WN11_'):
            src_path = os.path.join(src_folder, folder_name)
            dst_path = os.path.join(dst_folder, folder_name)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


if __name__=='__main__':

    LLMfold = './../LP_fb_wn_llama2/'
    # LLMfold = './../LP_fb_wn_chatglm2/'

    datasets_fb = []
    datasets_wn = []
    for name in os.listdir(LLMfold):
        if name.startswith('FB15k237_'):
            datasets_fb.append(name + '/')
        if name.startswith('WN18RR_'):
            datasets_wn.append(name + '/')
#     datasets_fb = ['FB15k237_srcdes_srckeywords_rel2des_struc_srckeywords_top3_selfloop_llama2/',
#                    'FB15k237_srcdes_srckeywords_struc_srckeywords_top3_selfloop_llama2/', ]

    for datasetname in datasets_fb:
        print(datasetname)
        CSprom_fb(datasetname)
    for datasetname in datasets_wn:
        print(datasetname)
        CSprom_wn(datasetname)

    for datasetname in datasets_fb:
        print(datasetname)
        LMKE_fb(datasetname)
    for datasetname in datasets_wn:
        print(datasetname)
        LMKE_wn(datasetname)

    SimKGC()
    
    LLMfold = './../TC_fb_wn_llama2/'
    TC()

