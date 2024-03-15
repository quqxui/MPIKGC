from read import read_ent2des, read_ent2expa_cot, read_ent2keyword, data_copy, change_triplefile_relation


################################################################
########################### Entity #############################
################################################################


def merge_srcdes_cotdes():
    dataset = f'{fb_or_wn}_srcdes_cotdes_'+ LLMname +'/'
    data_copy(datafold, dataset, fb_or_wn)
    ent2des = read_ent2des(datafold, fb_or_wn)
    ent2expa_cot = read_ent2expa_cot(LLMname, datafold,fb_or_wn, 'cotdes')
    if fb_or_wn in ['FB13','WN11']:
        filePath = 'entity2text.txt'
    elif fb_or_wn == 'FB15k237':
        filePath = 'FB15k_mid2description.txt'
    elif fb_or_wn == 'WN18RR':
        filePath = 'wordnet-mlj12-definitions.txt'
    with open(datafold + dataset + filePath,'w') as f:
        for entity in ent2des.keys():
            des = ent2des[entity] + " [SEP] " + ent2expa_cot[entity]
            f.write('{}\t{}\n'.format(entity, des))


def merge_srcdes_srckeywords():
    dataset = f'{fb_or_wn}_srcdes_srckeywords_'+ LLMname +'/'
    data_copy(datafold, dataset, fb_or_wn)
    ent2des = read_ent2des(datafold, fb_or_wn)
    ent2keyword = read_ent2keyword(LLMname, datafold,fb_or_wn, 'srckeywords')
    if fb_or_wn in ['FB13','WN11']:
        filePath = 'entity2text.txt'
    elif fb_or_wn == 'FB15k237':
        filePath = 'FB15k_mid2description.txt'
    elif fb_or_wn == 'WN18RR':
        filePath = 'wordnet-mlj12-definitions.txt'
    with open(datafold + dataset + filePath,'w') as f:
        for entity in ent2des.keys():
            des = ent2des[entity] + " [SEP] " + ent2keyword[entity]
            f.write('{}\t{}\n'.format(entity, des))


################################################################
########################### relation ###########################
################################################################

def merge_rel(rel_version):
    rel_version_str = '_'.join(rel_version)
    dataset = f'{fb_or_wn}_{rel_version_str}_{LLMname}/'
    data_copy(datafold, dataset, fb_or_wn)
    change_triplefile_relation(LLMname, datafold, dataset, fb_or_wn, rel_version)

    if fb_or_wn in ['FB13','WN11']:
        relation2text = {}
        relations = set()
        for file in ['train.tsv']:
            with open(datafold + dataset +file, 'r') as f:
                for line in f.readlines():
                    h,r,t  = line.strip('\n').split('\t')
                    relation2text[r] = r.replace('_',' ')
                    relations.add(r)
        with open(datafold + dataset + 'relation2text.txt', 'w') as f:
            for r,text in relation2text.items():
                f.write(f"{r}\t{text}\n")
        with open(datafold + dataset + 'relations.txt', 'w') as f:
            for r in relations:
                f.write(f"{r}\n")

################################################################
########################### structrue ##########################
################################################################


def merge_struc(top_loop):
    dataset = f'{fb_or_wn}_{top_loop}_{LLMname}/'
    data_copy(datafold, dataset, fb_or_wn)

    if fb_or_wn in ['FB13','WN11']:
        suffix = '.tsv'
    if fb_or_wn in ['FB15k237','WN18RR']:
        suffix = '.txt'
    with open(f'{datafold}{fb_or_wn}/struc/{top_loop}.txt', "r") as file1:
        with open(datafold + dataset + "train" +suffix, "a") as file2:
            lines = file1.readlines()
            for line in lines:
                file2.write(line)

    if fb_or_wn in ['FB13','WN11']:
        relation2text = {}
        relations = set()
        for file in ['train.tsv']:
            with open(datafold + dataset +file, 'r') as f:
                for line in f.readlines():
                    h,r,t  = line.strip('\n').split('\t')
                    relation2text[r] = r.replace('_',' ')
                    relations.add(r)
        with open(datafold + dataset + 'relation2text.txt', 'w') as f:
            for r,text in relation2text.items():
                f.write(f"{r}\t{text}\n")
        with open(datafold + dataset + 'relations.txt', 'w') as f:
            for r in relations:
                f.write(f"{r}\n")


################################################################
########################### combine ##########################
################################################################

def merge_srcdes_srckeywords_rel2sentence(rel_version):
    rel_version_str = '_'.join(rel_version)
    dataset = f'{fb_or_wn}_srcdes_srckeywords_{rel_version_str}_{LLMname}/'
    data_copy(datafold, dataset, fb_or_wn)
    ent2des = read_ent2des(datafold, fb_or_wn)
    ent2keyword = read_ent2keyword(LLMname, datafold,fb_or_wn, 'srckeywords')
    if fb_or_wn in ['FB13','WN11']:
        filePath = 'entity2text.txt'
    elif fb_or_wn == 'FB15k237':
        filePath = 'FB15k_mid2description.txt'
    elif fb_or_wn == 'WN18RR':
        filePath = 'wordnet-mlj12-definitions.txt'
    with open(datafold + dataset + filePath,'w') as f:
        for entity in ent2des.keys():
            des = ent2des[entity] + " [SEP] " + ent2keyword[entity]
            f.write('{}\t{}\n'.format(entity, des))

    change_triplefile_relation(LLMname, datafold, dataset, fb_or_wn, rel_version)

def merge_rel2sentence_struc_srckeywords_top3_selfloop(rel_version, top_loop):
    rel_version_str = '_'.join(rel_version)
    dataset = f'{fb_or_wn}_{rel_version_str}_{top_loop}_{LLMname}/'
    data_copy(datafold, dataset, fb_or_wn)

    change_triplefile_relation(LLMname, datafold, dataset, fb_or_wn, rel_version)

    if fb_or_wn in ['FB13','WN11']:
        suffix = '.tsv'
    if fb_or_wn in ['FB15k237','WN18RR']:
        suffix = '.txt'
    with open(f'{datafold}{fb_or_wn}/struc/{top_loop}.txt', "r") as file1:
        with open(datafold + dataset + "train" +suffix, "a") as file2:
            lines = file1.readlines()
            for line in lines:
                file2.write(line)


def merge_srcdes_srckeywords_rel2sentence_struc_srckeywords_top3_selfloop(rel_version, top_loop):
    rel_version_str = '_'.join(rel_version)
    dataset = f'{fb_or_wn}_srcdes_srckeywords_{rel_version_str}_{top_loop}_{LLMname}/'
    data_copy(datafold, dataset, fb_or_wn)
    ent2des = read_ent2des(datafold, fb_or_wn)
    ent2keyword = read_ent2keyword(LLMname, datafold,fb_or_wn, 'srckeywords')
    if fb_or_wn in ['FB13','WN11']:
        filePath = 'entity2text.txt'
    elif fb_or_wn == 'FB15k237':
        filePath = 'FB15k_mid2description.txt'
    elif fb_or_wn == 'WN18RR':
        filePath = 'wordnet-mlj12-definitions.txt'
    with open(datafold + dataset + filePath,'w') as f:
        for entity in ent2des.keys():
            des = ent2des[entity] + " [SEP] " + ent2keyword[entity]
            f.write('{}\t{}\n'.format(entity, des))

    change_triplefile_relation(LLMname, datafold, dataset, fb_or_wn, rel_version)

    if fb_or_wn in ['FB13','WN11']:
        suffix = '.tsv'
    if fb_or_wn in ['FB15k237','WN18RR']:
        suffix = '.txt'
    with open(f'{datafold}{fb_or_wn}/struc/{top_loop}.txt', "r") as file1:
        with open(datafold + dataset + "train" +suffix, "a") as file2:
            lines = file1.readlines()
            for line in lines:
                file2.write(line)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='description of the program')
    parser.add_argument( '--fb_or_wn',type=str,default='WN18RR') # 'FB15k237' 'WN11','FB13'
    parser.add_argument('--LLMname', type=str,default='llama2') # 'chatglm2'
    parser.add_argument('--datafold', type=str,default='./../LP_fb_wn_llama2/') # './../TC_fb_wn_llama2/' './../LP_fb_wn_llama2/' 
    args = parser.parse_args()
    datafold = args.datafold
    LLMname = args.LLMname
    fb_or_wn = args.fb_or_wn

    ################################################################
    ########################### Entity #############################
    ################################################################
    merge_srcdes_cotdes()
    merge_srcdes_srckeywords()

    ################################################################
    ########################### relation ###########################
    ################################################################
    rel_versions = [
        ['rel2des'],
        ['rel2sentence'],
        ['rel2reverse'],
        ['rel2des','rel2sentence'],
        ['rel2des','rel2reverse'],
        ['rel2sentence','rel2reverse'],
        ['rel2des', 'rel2sentence','rel2reverse'],
    ]
    for rel_version in rel_versions:
        merge_rel(rel_version)
    top_loop = 'struc_srckeywords_top3_selfloop'
    merge_srcdes_srckeywords_rel2sentence_struc_srckeywords_top3_selfloop(['rel2des'], top_loop)
    merge_srcdes_srckeywords_rel2sentence(['rel2des'])
    merge_rel2sentence_struc_srckeywords_top3_selfloop(['rel2des'], top_loop)

    ################################################################
    ########################### structrue ##########################
    ################################################################
    top_loops = [
        'struc_srckeywords_top1_selfloop','struc_srckeywords_top1',
        'struc_srckeywords_top2_selfloop','struc_srckeywords_top2',
        'struc_srckeywords_top3_selfloop','struc_srckeywords_top3',
        'struc_srckeywords_top4_selfloop','struc_srckeywords_top4',
        'struc_srckeywords_top5_selfloop','struc_srckeywords_top5']
    for top_loop in top_loops:
        merge_struc(top_loop)
    merge_struc('struc_srckeywords_top3_selfloop')

    ################################################################
    ########################### combine ##########################
    ################################################################
    merge_srcdes_srckeywords_rel2sentence_struc_srckeywords_top3_selfloop('struc_srckeywords_top3_selfloop')
