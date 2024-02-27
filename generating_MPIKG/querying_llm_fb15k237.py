from utils import log_output, mkdir_path, truncate, output_refine_llama2


class KG_fb():
    def __init__(self) -> None:
        self.rel2type, self.reltype = self.rel_type()
        self.ent2des = self.ent_des()
        self.ent2name = self.ent_name()
        self.h2rt, self.t2rh = self.ent_subgraph()
    def rel_type(self):
        ''' FB15k237 relation'''
        rel2type = {}
        reltype = set()
        for File in ['train.txt','test.txt','valid.txt']:
            with open(f'{LLMfold}FB15k237/'+File,'r') as f:
                for line in f.readlines():
                    h,r,t = line.strip().split('\t')
                    rel2type[r] = r.split('/')[1]
                    reltype.add(r.split('/')[1])
        return rel2type, reltype

    def ent_subgraph(self):
        from collections import defaultdict
        h2rt = defaultdict(list)
        t2rh = defaultdict(list)
        for File in ['train.txt','valid.txt']:
            with open(f'{LLMfold}FB15k237/'+File,'r') as f:
                for line in f.readlines():
                    h,r,t = line.strip().split('\t')
                    text_h = self.ent2name[h]
                    text_r = self.rel2type[r]
                    text_t = self.ent2name[t]
                    h2rt[text_h].append((text_r,text_t))
                    t2rh[text_t].append((text_r,text_h))
        # print(len(h2rt),len(t2rh)) # 13822 13435
        # print(h2rt['/m/06rf7'])
        return h2rt, t2rh
    
    def ent_des(self):
        ent2des = {}
        with open(f'{LLMfold}FB15k237/FB15k_mid2description.txt','r') as f:
            for line in f.readlines():
                ent,des = line.strip().split('\t')
                ent2des[ent] = des
        return ent2des
    
    def ent_name(self):
        ent2name = {}
        with open(f'{LLMfold}FB15k237/FB15k_mid2name.txt','r') as f:
            for line in f.readlines():
                ent,name = line.strip().split('\t')
                ent2name[ent] = name.replace("_", " ")
        return ent2name


def Ent2NameDes(path):
    ent2name = {}
    with open(path,'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,name = line.strip().split('\t')
            ent2name[ent] = name.replace("_", " ")
    ent2des_all = {}
    with open(f'{LLMfold}FB15k237/FB15k_mid2description.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,des = line.strip().split('\t')
            ent2des_all[ent] = des
    return ent2name, ent2des_all


def expansion_cot_fb(args, llm):
    ent2name, _ = Ent2NameDes(f'{LLMfold}FB15k237/FB15k_mid2name.txt')
    save_path = f'{LLMfold}FB15k237/cotdes.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for ent,name in ent2name.items():
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. If you don't know the answer to a question, just replay: I don't know. <</SYS>> \n" + \
                            f"Please provide all informations about " + name + ". Give the rationale before answering:[/INST] "
            elif LLMname == 'ChatGLM2':
                template =  f"Please provide all informations you know about " + name 
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)



def keyword_fb(args, llm):
    ent2expansion = {}
    with open(f'{LLMfold}FB15k237/cotdes.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,expansion = line.strip().split('\t')
            if LLMname == 'LLAMA2':
                expansion = output_refine_llama2(expansion)
            ent2expansion[ent] = expansion
    save_path = f'{LLMfold}FB15k237/cotkeywords.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for ent,expansion  in ent2expansion.items():
            expansion = truncate(expansion,200)
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n  Please extract the five most representative keywords from the following text: <</SYS>> \n" + \
                            f"text: " + expansion + " \n keywords:[/INST] "
            elif LLMname == 'ChatGLM2':
                template = 'Please extract the five most representative keywords from the following text: ' + expansion + ' \n keywords:'
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)
            

def keyword_fb_src(args, llm):
    ent2name, ent2des_all = Ent2NameDes(f'{LLMfold}FB15k237/FB15k_mid2name.txt')
    save_path = f'{LLMfold}FB15k237/srckeywords.txt'
    with open(save_path , 'w', encoding='utf-8') as f:
        for ent,name in ent2name.items():
            des_all = ent2des_all[ent]
            des_all = truncate(des_all,200)
            if LLMname == 'LLAMA2':
                template = f"<s>[INST] <<SYS>> \n  Please extract the five most representative keywords from the following text: <</SYS>> \n" + \
                            f"text: " + des_all + " \n keywords:[/INST] "
            elif LLMname == 'ChatGLM2':
                template = 'Please extract the five most representative keywords from the following text: ' + des_all + " \n keywords:"
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(ent,ans))
            log_output(ans)



def relation_des_fb(args, llm):
    rel2text = {}
    with open(f'{LLMfold}FB15k237/train.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            text = ' '.join(r.split('/')) 
            rel2text[r] = text 

    # rel2des
    save_path = f'{LLMfold}FB15k237/rel2des.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for r,text in rel2text.items():
            if LLMname == 'LLAMA2':
                template =  f"<s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. <</SYS>> \n" + \
                        f"Please provide an explanation of the significance of the relation {text} in a knowledge graph with one sentence. [/INST]"
            elif LLMname in ['ChatGLM2','GPT']:
                template = f"Please provide an explanation of the significance of the relation {text} in a knowledge graph with one sentence."
            ans = llm.qa(template)
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            f.write('{}\t{}\n'.format(r, ans))
            log_output(ans)
    
    # rel2sentence
    save_path = f'{LLMfold}FB15k237/rel2sentence.txt'
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
    save_path = f'{LLMfold}FB15k237/rel2reverse.txt'
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

    relation_des_fb(args, llm)
    expansion_cot_fb(args, llm)
    keyword_fb_src(args, llm)
    keyword_fb(args, llm)
    


