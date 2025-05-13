# Multi-perspective Improvement of Knowledge Graph Completion with Large Language Models

This is the code and data of paper **Multi-perspective Improvement of Knowledge Graph Completion with Large Language Models**, LREC-COLING 2024.

The entire project consists of three steps: (1) Generating data, (2) Processing data and (3) Runing KGC models.

##  Generating data
You can directly download the data we generated from [Goolge Drive](https://drive.google.com/drive/folders/1pBXBB0PcNOpfIZETSZCD4QI61L1-JzN_?usp=sharing). Put the data in current directory, e.g., `MPIKGC/LP_fb_wn_chatglm2/`.

 If you want to generate data with LLMs by yourself, you can run the following command to query LLMs for data:

First, `cd generating_MPIKG/`, then

For FB15k237:
```python 
python querying_llm_fb15k237.py --max_length 256 --temperature 0.2 --cuda 0 --batchsize 1 --LLMfold './../LP_fb_wn_chatglm2/' --LLMname ChatGLM2
```
For WN18RR:
```python 
python querying_llm_wn18rr.py --max_length 256 --temperature 0.2 --cuda 0 --batchsize 1 --LLMfold './../LP_fb_wn_chatglm2/' --LLMname ChatGLM2
```

For FB13 and WN11:
```python 
python querying_llm_TC_FB13_WN11.py --max_length 256 --temperature 0.2 --cuda 0 --batchsize 1 --LLMfold './../TC_fb_wn_llama2/' --LLMname ChatGLM2 --fb_or_wn FB13
```

After getting extracted keywords, you can genearate the top K matching entities for Structure Extraction, and change the default parameters in codes to obtain the desired data:
```python
python textmatch4structure.py 
```

Finally, generating data in corresponding path, e.g., `LP_fb_wn_chatglm2/FB15k237/`:

`cotdes.txt` for `MPIKGC-E`

`rel2des.txt` for `MPIKGC-R Global` 

`rel2sentence.txt` for `MPIKGC-R Local` 

`rel2reverse.txt` for `MPIKGC-R Reverse` 

`struc/*` for `MPIKGC-S`


## Processing data
Merge the enhanced data with original KG to generate new KG with regular form:
```python
cd processing_MPIKG
python merge.py 
```
Download KGC models from their repository: [CSprom-KG](https://github.com/chenchens190009/CSProm-KG),
[LMKE](https://github.com/Neph0s/LMKE), 
[SimKGC](https://github.com/intfloat/SimKGC), 
[KG-BERT](https://github.com/yao8839836/kg-bert). Put these project in current directory, e.g., `MPIKGC/CSprom-KG/`.


For diffrent KGC models, you need to slightly modify the form of data by runing:
```python
python data4kgc_models.py 
```
This code will transfer the form of data to adapt to different KGC model requirements, and copy the data to corresponding model project folder.

Note that KG-BERT need same form as LMKE, You can directly copy it over.
## Runing KGC models
Configure according to the environment and process requirements of each model. Hyperparameters for KGC models can be found in Appendices of our paper.

## Additional Notes
Since this implementation involves 5 prompts augmentation + 4 KGC models + 4 datasets, the merging code is relatively complex.
If you only wish to implement our method on your own dataset, you do not need to reference our code. Instead, you can follow the prompts provided in the paper to generate data independently. The main focus of the project code is on data processing.

