# -*- coding: utf-8 -*-
import re
import sys
sys.path.append("..")
from src.read import read_ent2keyword
import tqdm

def WN11_ent2keywords():
    ent2keywords = {}
    with open('./../TC_fb_wn/WN11/entity2text.txt','r') as f:
        for line in f.readlines():
            ent, text  = line.strip().split('\t')
            ent2keywords[ent] = text
    return ent2keywords
def word_matching(datafold, LLMname,fb_or_wn ,src_or_expa):
    if fb_or_wn =='FB15k237':
        sameas = 'same/as'
    elif fb_or_wn in ['WN18RR','WN11','FB13']:
        sameas = 'same_as'
    
    if fb_or_wn == "WN11":
            ent2keywords = WN11_ent2keywords()
    else:
        ent2keywords = read_ent2keyword(LLMname, datafold, fb_or_wn, src_or_expa)
    
    ent2senset = {}
    for ent in ent2keywords.keys():
        sen = re.sub(r"[^a-zA-Z0-9]+", " ", ent2keywords[ent]).lower().split()
        sen = set(sen)
        ent2senset[ent] = sen

    
    f_top1 = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top1.txt','w')
    f_top2 = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top2.txt','w')
    f_top3 = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top3.txt','w')
    f_top4 = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top4.txt','w')
    f_top5 = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top5.txt','w')

    f_top1_selfloop = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top1_selfloop.txt','w')
    f_top2_selfloop = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top2_selfloop.txt','w')
    f_top3_selfloop = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top3_selfloop.txt','w')
    f_top4_selfloop = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top4_selfloop.txt','w')
    f_top5_selfloop = open(datafold + f'{fb_or_wn}/struc/struc_{src_or_expa}_top5_selfloop.txt','w')

    for ent1 in tqdm.tqdm(ent2keywords.keys()):
        ent22score = {}
        for ent2 in ent2keywords.keys():
            if ent2 == ent1 : continue
            same = ent2senset[ent1] & ent2senset[ent2]
            if len(same):
                similarity_score = max(len(same) / len(ent2senset[ent1]), len(same) / len(ent2senset[ent2]))
                ent22score[ent2] = similarity_score
        sorted_dict = dict(sorted(ent22score.items(), key=lambda x: x[1], reverse=True))

        for count, (ent2,_) in enumerate(sorted_dict.items()):
            if count >=5: break
            if count <1:
                f_top1.write(f"{ent1}\t{sameas}\t{ent2}\n")
            if count <2:
                f_top2.write(f"{ent1}\t{sameas}\t{ent2}\n")
            if count <3:
                f_top3.write(f"{ent1}\t{sameas}\t{ent2}\n")
            if count <4:
                f_top4.write(f"{ent1}\t{sameas}\t{ent2}\n")
            if count <5:
                f_top5.write(f"{ent1}\t{sameas}\t{ent2}\n")

        f_top1_selfloop.write(f"{ent1}\t{sameas}\t{ent1}\n")
        f_top2_selfloop.write(f"{ent1}\t{sameas}\t{ent1}\n")
        f_top3_selfloop.write(f"{ent1}\t{sameas}\t{ent1}\n")
        f_top4_selfloop.write(f"{ent1}\t{sameas}\t{ent1}\n")
        f_top5_selfloop.write(f"{ent1}\t{sameas}\t{ent1}\n")

    f_top1.close()
    f_top2.close()
    f_top3.close()
    f_top4.close()
    f_top5.close()
    f_top1_selfloop.close()
    f_top2_selfloop.close()
    f_top3_selfloop.close()
    f_top4_selfloop.close()
    f_top5_selfloop.close()

if __name__ == '__main__':
    fb_or_wn = 'WN18RR' # 'FB15k237' 'WN11','FB13'
    LLMname = 'llama2'# 'chatglm2'
    src_or_expa = 'srckeywords', # 'cotkeywords'
    datafold = './../LP_fb_wn_llama2/' # './../TC_fb_wn_llama2/' './../LP_fb_wn_llama2/'  
    word_matching(LLMname,fb_or_wn ,src_or_expa)



