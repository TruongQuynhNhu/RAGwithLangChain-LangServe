# RAGwithLangChain-LangServe
Build Q&amp;A application with RAG using LangChain and LangServe

Reference: https://python.langchain.com/docs/

## 1. Installation
(1) install all the necessary package for the project mention infile `requirements.txt`

```
pip install -r requirements.txt
```
(2) run file `serve.py`

```
python serve.py
```
after filling HUGGINGFACE API and OPENAI KEY. the app is deployed on http://localhost:8000/agent/playground/

## 2. Results

![image](https://github.com/TruongQuynhNhu/RAGwithLangChain-LangServe/assets/107611691/54d11ae0-c60e-4c7a-8628-84e78d38b350)
![image](https://github.com/TruongQuynhNhu/RAGwithLangChain-LangServe/assets/107611691/04834e9f-5de7-46d4-9e28-c205c8a790fc)
![image](https://github.com/TruongQuynhNhu/RAGwithLangChain-LangServe/assets/107611691/d278c111-7b3f-4f17-b432-534d0fa4fb94)

## 3. Some Reflections

- With simple, intuitive questions, the answers are usually quite good and acceptable. Questions that require multi-layered reasoning have not yet received a suitable answer.

- For testing purposes, the app is currently using a quantized model. Therefore, the ability to answer a complete and appropriate sentence is not yet available. => You can try with the initial model () that has not been quantized when there are conditions of computing resources and time.

## 4. Citation

```
https://python.langchain.com/docs/
```
