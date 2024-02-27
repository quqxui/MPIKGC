from utils import log_output, mkdir_path, output_refine_llama2

class KG_wn():
    def __init__(self) -> None:
        self.h2rt, self.t2rh = self.ent_subgraph()

    def ent_subgraph(self):
        from collections import defaultdict
        h2rt = defaultdict(list)
        t2rh = defaultdict(list)
        for File in ['train.txt','valid.txt']:
            with open(f'{LLMfold}WN18RR/'+File,'r', encoding='utf-8') as f:
                for line in f.readlines():
                    h,r,t = line.strip().split('\t')
                    text_h = self.ent2name[h]
                    text_r = ' '.join(r.split('_'))
                    text_t = self.ent2name[t]
                    h2rt[text_h].append((text_r,text_t))
                    t2rh[text_t].append((text_r,text_h))
        return h2rt, t2rh


def Ent2NameDes(path):
    ent2name = {}
    ent2des = {}
    with open(path,'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent, name, des = line.strip().split('\t')
            ent2name[ent] =  name.split('_')[2]
            ent2des[ent] = des
    return ent2name, ent2des


def expansion_cot_wn(args,llm):
    # extend of LLM
    ent2name, ent2des = Ent2NameDes(f'{LLMfold}WN18RR/wordnet-mlj12-definitions.txt')

    save_path = f'{LLMfold}WN18RR/cotdes.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for ent,name in ent2name.items():
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. If you don't know the answer to a question, just replay: I don't know. <</SYS>> \n" + \
                            f"Please explain the meaning of " + name + ". Give the rationale before answering:[/INST] "
            elif LLMname == 'ChatGLM2':
                template = f"Please explain the meaning of " + name   + ". Give the rationale before answering: "
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)
            

def keyword_wn(args, llm):
    ent2expansion = {}
    with open(f'{LLMfold}WN18RR/cotdes.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,expansion = line.strip().split('\t')
            if LLMname == 'LLAMA2':
                expansion = output_refine_llama2(expansion)
            ent2expansion[ent] = expansion

    save_path = f'{LLMfold}WN18RR/cotkeywords.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for ent,expansion in ent2expansion.items():
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n  Please extract the five most representative keywords from the following text: <</SYS>> \n" + \
                            f"text: " + expansion + " \n keywords:[/INST] "
            elif LLMname == 'ChatGLM2':
                template = f"Please extract the five most representative keywords from the following text: " + expansion + " \n keywords: "

            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)
            

def keyword_wn_src(args, llm):
    ent2name, ent2des = Ent2NameDes(f'{LLMfold}WN18RR/wordnet-mlj12-definitions.txt')

    save_path = f'{LLMfold}WN18RR/srckeywords.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for ent, _ in ent2name.items():
            des = ent2des[ent]
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n  Please extract the two most representative keywords from the following text: <</SYS>> \n" + \
                            f"text: " + des + " \n keywords:[/INST] "
            elif LLMname == 'ChatGLM2':
                template = "Please extract the two most representative keywords from the following text: " + des +" \n keywords:"
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)

def relation_des_wn(args, llm):
    rel2text = {}
    with open(f'{LLMfold}WN18RR/train.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            text = ' '.join(r.split('_')) 
            rel2text[r] = text 

    # rel2des
    save_path = f'{LLMfold}WN18RR/rel2des.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for r,text in rel2text.items():
            if LLMname == 'LLAMA2':
                template =  f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. <</SYS>> \n" + \
                        f"Please provide an explanation of the significance of the relation {text} in a knowledge graph with one sentence. [/INST]"
            elif LLMname == 'ChatGLM2':
                template = f"Please provide an explanation of the significance of the relation {text} in a knowledge graph with one sentence."
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(r, ans))
            log_output(ans)
    
    # rel2sentence
    save_path = f'{LLMfold}WN18RR/rel2sentence.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for r,text in rel2text.items():
            if LLMname == 'LLAMA2':
                template =  f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. <</SYS>> \n" + \
                        f"Please provide an explanation of the meaning of the triplet (head entity, {text}, tail entity) and rephrase it into a sentence. [/INST]"
            elif LLMname == 'ChatGLM2':
                template = f"Please provide an explanation of the meaning of the triplet (head entity, {text}, tail entity) and rephrase it into a sentence."
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(r, ans))
            log_output(ans)

    # rel2reverse
    save_path = f'{LLMfold}WN18RR/rel2reverse.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for r,text in rel2text.items():
            if LLMname == 'LLAMA2':
                template =  f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. <</SYS>> \n" + \
                        f"Please convert the relation {text} into a verb form and provide a statement in the passive voice. [/INST]"
            elif LLMname == 'ChatGLM2':
                template = f"Please convert the relation {text} into a verb form and provide a statement in the passive voice. "
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(r, ans))
            log_output(ans)

if __name__=='__main__':

    import argparse
    from LLMModels import LLAMA2, ChatGLM2, GPT
    parser = argparse.ArgumentParser(description='description of the program')
    parser.add_argument( '--cuda',type=str)
    parser.add_argument('--max_length', type=int, help='max length of LLMs')
    parser.add_argument('--temperature', type=float, help='temperature of LLMs')
    parser.add_argument('--batchsize', type=int, help='batchsize')
    parser.add_argument( '--LLMfold',type=str, help='The path of dataset') # './../LP_fb_wn_llama2/' 
    parser.add_argument( '--LLMname',type=str, help='The name of LLMs') # LLAMA2  ChatGLM2 GPT

    args = parser.parse_args()

    LLMfold = args.LLMfold
    LLMname = args.LLMname
    if LLMname=='LLAMA2':
        llm = LLAMA2(args)
    elif LLMname=='ChatGLM2':
        llm = ChatGLM2(args)
    elif LLMname== 'GPT':
        llm = GPT(args)

    relation_des_wn(args, llm)
    expansion_cot_wn(args, llm)
    keyword_wn_src(args, llm)
    keyword_wn(args, llm)

