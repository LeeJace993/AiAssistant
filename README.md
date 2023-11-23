# 项目介绍
本项目最终实现三大功能:健康状态分析 问题咨询 舌部诊断
* 健康状态分析：通过建立COT以及Template训练OpenAI LLM模型，根据本地database中的人体各项健康指标.docx文件建立Chroma向量数据库。利用本地向量数据库回答用户输入Query
* 问题咨询:通过建立COT以及Template训练OpenAI LLM模型，根据本地database中的AI_assistant.docx文件建立Chroma向量数据库。利用本地向量数据库回答用户输入Query
* 舌部诊断:通过Unet网络训练提取出舌头的语义类别，再通过支持向量机的方法训练舌部颜色识别模型，模型返回舌苔以及舌质的颜色。最终将获取的结果输入OpenAI LLM模型实现舌部诊断。
* Streamlit开源访问：浏览器输入 https://aiassistant-tsinghua.streamlit.app/ 即可访问
# 使用方法
## 1.安装项目依赖
`pip install -r requirements.txt`
## 2.训练Unet图像分割网络
`python TongueSegmentation.py`  
该代码将利用TongeImageDataset中的Image以及Mask训练Unet网络，并将训练好的模型存储到utils中  
model_2_1.model即是训练好的一个模型
## 3.训练SVM向量机模型 
`python model_pickle.py`   
该代码将利用ColorReference中的颜色信息训练SVM模型，最终将训练好的分类器保存在 utils中prediction_test_classifier.pkl 即使训练好的一个分类器
## 4.运行整个项目
`streamlit run main.py`   
在streamlit上运行本项目
# 注意
1.测试时请使用自己的OpenAI Key 在main.py第22行修改  
2.LangChain问答需要连接VPN
