from utils import log_output, mkdir_path, truncate


def Ent2NameDes(path):
    ent2name = {}
    with open(path,'r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,name = line.strip().split('\t')
            ent2name[ent] = name
    ent2des_all = {}
    with open(f'{LLMfold}{fb_or_wn}/entity2text.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            ent,des = line.strip().split('\t')
            ent2des_all[ent] = des
    return ent2name, ent2des_all


def expansion_cot_fb(args, llm):
    ent2name, _ = Ent2NameDes(f'{LLMfold}{fb_or_wn}/entity2textshort.txt')
    save_path = f'{LLMfold}{fb_or_wn}/cotdes.txt'
    with open(save_path , 'w', encoding='utf-8') as f:
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


def keyword_fb_src(args, llm):
    ent2name, ent2des_all = Ent2NameDes(f'{LLMfold}{fb_or_wn}/srckeywords.txt')
    
    save_path = f'{LLMfold}{fb_or_wn}/srckeywords.txt'
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
    with open(f'{LLMfold}{fb_or_wn}/relations.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            r = line.strip()
            text = ' '.join(r.split('_')) 
            rel2text[r] = text

    save_path = f'{LLMfold}{fb_or_wn}/rel2des.txt'
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
    
    save_path = f'{LLMfold}{fb_or_wn}/rel2sentence.txt'
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

    save_path = f'{LLMfold}{fb_or_wn}/rel2reverse.txt'
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
    parser.add_argument( '--LLMfold',type=str, help='The path of dataset') # './../TC_fb_wn_llama2/' 
    parser.add_argument( '--LLMname',type=str, help='The name of LLMs') # LLAMA2  ChatGLM2 GPT
    parser.add_argument( '--fb_or_wn',type=str, help='dataset') # FB13  WN11

    args = parser.parse_args()

    LLMfold = args.LLMfold
    LLMname = args.LLMname
    fb_or_wn = args.fb_or_wn
    
    if LLMname=='LLAMA2':
        llm = LLAMA2(args)
    elif LLMname=='ChatGLM2':
        llm = ChatGLM2(args)
    elif LLMname== 'GPT':
        llm = GPT(args)

    
    keyword_fb_src(args, llm)
    expansion_cot_fb(args, llm)
    relation_des_fb(args, llm)


