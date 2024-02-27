
import torch
import transformers
import openai

class ChatGLM2():
    def __init__(self,args) -> None:
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b", trust_remote_code=True,revision="v1.0")
        self.model = AutoModel.from_pretrained("chatglm2-6b", trust_remote_code=True,revision="v1.0",device="cuda:"+args.cuda,)#.to(args.device)
        self.args = args
    def qa(self,prompt):
        with torch.no_grad():
            if isinstance(prompt, list): # batch
                answers = []
                for pr in prompt:
                    ans, history = self.model.chat(self.tokenizer, pr, history=[], max_length=self.args.max_length,temperature=self.args.temperature)
                    ans = ans.replace('\n', ' ')
                    ans = ans.replace('\t', ' ')
                    answers.append(ans)
                return answers
            else:
                ans, history = self.model.chat(self.tokenizer, prompt, history=[])
                ans = ans.replace('\n', ' ')
                ans = ans.replace('\t', ' ')
                return ans


class GPT():
    def __init__(self,args) -> None:  
        
        openai.api_key = "Your API Key"
        self.args = args
    def qa(self,prompt):
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=self.args.temperature,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],)
        ans = completion.choices[0].message.content
        ans = ans.replace('\n', ' ')
        ans = ans.replace('\t', ' ')
        return ans

class LLAMA2():
    def __init__(self,args) -> None:
        from transformers import AutoTokenizer
        model_path = 'LLaMA-2-7b-chat-hf'
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float16,
            # device_map="auto",#"auto",
            device="cuda:"+args.cuda,
        )

    def qa(self,prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.args.max_length,
            temperature=self.args.temperature,
        )
        if isinstance(prompt, list): # batch
            answers = []
            for seq in sequences:
                ans = seq[0]['generated_text']
                ans = ans.replace('\n', ' ')
                ans = ans.replace('\t', ' ')
                answers.append(ans)
            return answers
        else:
            ans = sequences[0]['generated_text']
            ans = ans.replace('\n', ' ')
            ans = ans.replace('\t', ' ')
            return ans

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='description of the program')
    parser.add_argument( '--cuda',default='1', type=str, help='help message for arg1')
    parser.add_argument('--split',default=1, type=int, help='help message for arg2')
    parser.add_argument('--max_length',default=200, type=int, help='help message for arg2')
    parser.add_argument('--temperature', default=0.2,type=float, help='help message for arg2')
    args = parser.parse_args()

    llm = ChatGLM2(args)
    ans = llm.qa(['Tell me all you know about trump:',
                  'Tell me all you know about Biden:',])
    print(ans)

    ans = llm.qa('Tell me all the information about Biden:',)
    print(ans)