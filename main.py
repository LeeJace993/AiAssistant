from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
## from langchain import OpenAI, VectorDBQA,PromptTemplate
from langchain import OpenAI, VectorDBQA, PromptTemplate
from langchain.chat_models import ChatOpenAI


from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
import streamlit as st
import openai


from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img
from keras.preprocessing import image

import os
import detact_run
from PIL import Image

import nltk
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'tokenizers/{pkg}') if "punkt" in pkg else nltk.data.find(f'taggers/{pkg}')
    except LookupError:
        nltk.download(pkg)

#请输入你的OPENAI_API_KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 检查是否加载成功
if not OPENAI_API_KEY:
    st.error("⚠️ 未检测到 OpenAI API Key，请检查 Streamlit Secrets 设置！")
else:
    st.success("✅ 成功加载 OpenAI API Key！")
template1="""
    你是一名老年人的健康助手，负责给你的用户提供一些健康相关的建议。
    接下来会有四个种类的问题，分别是健康建议、医疗信息、紧急救助和心理健康。
    其中健康建议包含饮食、锻炼、睡眠和健康监测四个部分。
    医疗信息包含了老年人常见疾病以及相应的药物治疗两部分。紧急救助包含急救指南和急救设备的使用方案两部分。
    心理健康包含老年人心理的一般特点、老年人常见的心理表现和心理问题和老年人心理问题预防三部分。
    这四个种类分别可能的问题如下：
    健康建议：
    我应该如何定制一个适合糖尿病患者的健康的一日三餐食谱？
    我是一个50岁且腿脚不便的糖尿病患者,我该如何选择适合自己的锻炼方式？
    我最近晚上睡眠不足，如何提高我的睡眠质量？
    医疗信息：
    如何预防和控制高血压？
    我是一名糖尿病患者，有没有什么推荐使用的药物？在使用药物时有没有注意事项？
    紧急救助：
    在心肺复苏过程中，我应该如何操作？
    如何正确使用自动体外除颤器(AED)?
    心理健康：
    我应该如何关注自己的心理健康？
    如何识别和应对焦虑症状？
    下面是问题的种类
    种类: {type}
    你的回答: 
"""
template2="""
你是一名老年人的健康助手，负责根据用户提供的信息分析用户的健康状况。
用户会给你提供5个方面的信息,分别是体温、心率、血压、血糖、血脂
用户提供的信息是数值，你需要将用户提供的数值和标准范围的数值进行比较，判断用户相应指标是否正常，如果在标准范围即为正常，不在标准范围则不正常，需要对用户可能存在的健康问题进行分析，并给出合理的治疗方案，建议用户及时就医。
根据提供的5个数据综合分析用户的健康情况。
用户提供的信息可能如下：
体温{temperature}:36.5℃
心率{heart_rate}74/min
血压{blood_pressure}110mmHg 86mmHg
血糖{blood_sugar}4.5mmol/L
血脂{blood_fat}3.7mmol/L
"""
def load_LLM():
    """加载大语言模型"""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    return llm
    # COT prompt for diabetes-related advice - divided into smaller parts
cot_part1 = """
1、提问：我是一名糖尿症患者，我在日常生活中有哪些需要注意的事情？
回答：因为你是一名糖尿症患者，我会根据健康建议、医疗信息、紧急救助和心理健康四个方面给出我的回答。
首先是健康建议：
糖尿病患者在日常饮食中供能营养素应以碳水化合物（50%~55%）为主，宜多选择能量密度高且富含膳食纤维、低升血糖指数的食物，增加蔬菜和适当比例的低糖水果。蛋白摄入建议为1.0~1.5 g·kg-1·d-1，以优质蛋白为主，可改善胰岛素抵抗、减轻年龄相关的肌肉减少等。同时注重适当锻炼、保证充足的睡眠、定期监测血糖变化。
然后是医疗信息：
糖尿病症状包括“三多一少”，即多饮、多尿、多食、体重减轻。病情加重时，可能导致并发症，如视网膜病变、肾病、神经病变等。
糖尿病与遗传、肥胖、饮食不当、运动不足、年龄增长等有关。
糖尿病的治疗为药物治疗（如胰岛素、降糖药等）+ 改善生活方式（低糖饮食、适量运动、戒烟限酒等）。
建议患者定期检查血糖，及时发现和控制糖尿病。
其次是紧急救助：
由于患者日常进行胰岛素的注射，以下是胰岛素泵的使用步骤：
患者确诊为糖尿病。
医生评估患者适合使用胰岛素泵。
选择合适的胰岛素泵品牌和型号。
医生为患者安装胰岛素泵，并指导患者如何操作。
患者按照医生建议调整胰岛素剂量，监测血糖水平。
定期与医生沟通，根据血糖控制情况调整胰岛素剂量和泵设置。
了解胰岛素泵的维护方法，如清洁、避免潮湿等。
最后是心理健康：
糖尿病为中老年常见疾病，患糖尿病不必太过担忧，否则思虑过多,心情抑郁,反而容易降低对疾病的抵抗力。有了病首先是不要紧张,要镇静地面对现实积极治疗。过去认为是不治之症的癌症,只要早期发现,也有较多办法对它进行控制了。糖尿病只要按时服药,注意起居有节律,避免情绪波动,同样可以缓解，不会影响日常生活。

    """

cot_part2 = """
        提问：我应该如何定制一个适合糖尿病患者的健康的一日三餐食谱？
回答：
定制适合糖尿病患者的一日三餐食谱时，需要遵循以下原则：
控制总热量：根据患者的体重、劳动强度、血糖情况、有无并发症及用药情况等因素，确定总热量指标，并随时调整、灵活掌握。
均衡营养：确保膳食中富含蛋白质、脂肪、碳水化合物、膳食纤维、维生素和矿物质等营养成分。
低糖、低脂、低盐：减少糖、油脂和盐的摄入，以降低血糖、血压和血脂升高的风险。
升糖指数低的食物：选择升糖指数（GI）较低的食物，如粗粮、蔬菜和水果等，有助于稳定血糖。
优质蛋白：优先选择富含优质蛋白的食物，如瘦肉、鱼类、牛奶等。
增加纤维摄入：适量增加膳食纤维摄入，有助于改善血糖控制和降低胆固醇。
以下是一个参考的适合糖尿病患者的一日三餐食谱：
1、早餐+水果
主食:高纤维馒头或饼等高纤维主食
副食:煮鸡蛋或荷包蛋一个。淡豆浆、牛奶或小米粥可任选一种。凉拌蔬菜。
饭后过一段时间后可以吃水果,但是不要吃的很多。
2、午餐
主食:高纤维大米饭、高纤维馒头、高纤维面条或其它高纤维主食
副食:瘦肉、鱼、鸡、鸭可根据情况选择。清炒蔬菜、凉拌蔬菜、豆制品等。
3、晚餐
主食:高纤维馒头、高纤维大米饭等高纤维主食。喜欢喝粥者可根据个人习惯选择小米粥、绿豆粥、红小豆粥等。
副食:蔬菜、豆制品等。鸡、鸭、肉、鱼等可根据个人喜爱情况选择。
4、夜宵
晚上睡觉前喝鲜纯牛奶300毫升,约一杯。
5、其他饮食推荐
鳝鱼：患有高血糖的患者在平时的生活中应该适当的多吃些鳝鱼,这种食物具有很好的降血糖功效,这是由于在鳝鱼中含有大量丰富的“黄鳝鱼素a”和“黄鳝鱼素b”,这两种物质可有效的帮助我们人体恢复调节血糖正常的生理功能。
洋葱：洋葱是我们日常生活中最常见的一种蔬菜,我国中医指出洋葱性微温,与葱、蒜性味相近,经常食用具有很好的健胃、增进食欲、行气宽中的功效。如果与大蒜一块食用的话,则具有很好的降血糖功效,这是由于在洋葱中含有类似降糖药物甲磺丁脲,经常食用不但可以有效的降血糖,同时还具有很好的饱腹作用,是糖尿病患者理想的食品。
魔芋：经过研究发现,魔芋是一种低热能、高纤维素食物,经常食用不但不用担心虎发胖,同时还具有很好的减肥功效。而且魔芋中不仅含有大量的营养物质,同时其医疗保健价值也非常的高。在魔芋中含有一种葡萄甘露聚糖物质,这种营养物质对降低糖尿病患者的血糖有较好的效果。
注意：以上食谱仅供参考，具体食用量需根据个人情况和医生建议调整。同时，糖尿病患者在饮食控制的基础上，还需注意运动、药物、监测血糖等方面的综合管理。

    """


    # Combine the smaller parts of COT with your large language model
combined_prompt = f"{template1}\n{cot_part1}\n{cot_part2}"
prompt1 = PromptTemplate(
    input_variables=["type"],
    template=combined_prompt
)
# 加载您的大型语言模型
llm=load_LLM()
loader1 = UnstructuredWordDocumentLoader("./database/AI_assistant.docx")
document1=loader1.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts=text_splitter.split_documents(document1)
embeddings1=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch1=Chroma.from_documents(texts,embeddings1) # TO  create a Vectorbase
qa1 = VectorDBQA.from_chain_type(
    llm=llm,                   # 用你上面刚加载好的 ChatOpenAI 对象
    chain_type='refine',
    vectorstore=docsearch1
)

prompt1=PromptTemplate(
    input_variables=["type"],
    template=template1,
)
prompt2=PromptTemplate(
    input_variables=["temperature","heart_rate","blood_pressure","blood_sugar","blood_fat"],
    template=template2,
)
def load_LLM2():
    """加载第二个语言模型"""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    return llm

llm=load_LLM2()
loader2 = UnstructuredWordDocumentLoader("./database/人体各项健康指标.docx")
document2=loader2.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts=text_splitter.split_documents(document2)
embeddings2=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch2=Chroma.from_documents(texts,embeddings2) # TO  create a Vectorbase
qa2 = VectorDBQA.from_chain_type(
    llm=llm, 
    chain_type='refine',
    vectorstore=docsearch2
)



st.set_page_config(page_title="老年智能助手",page_icon=":robot:")
st.header("老年智能助手")
col1,col2=st.columns(2)

with col1:
    st.markdown("专门为老年人设计的智能助手,此智能助手由[LangChain](www.langchain.com) 以及 [OpenAI](https://openai.com)驱动。  "    
    )  
    st.markdown("*第16组 殷亦达 张泽艺*")
    st.markdown("本助手有 健康状态分析 问题咨询 舌部诊断三大功能，用户可以通过页面左边的导航栏切换不同功能")

with col2:
    st.image(image="./utils/R-C.jpeg",width=300)
    
    
def page_home():
    st.title('健康状态分析')
    # 在Home页面中显示数据和功能组件
    st.markdown("请填入您的基本信息")
    # 添加输入框来收集用户数据
    body_temperature = st.number_input("体温（摄氏度）", min_value=0.0, max_value=100.0, value=37.0)
    heart_rate = st.number_input("心率（每分钟）", min_value=0, max_value=300, value=75)
    blood_pressure = st.number_input("血压（mmHg）", min_value=0, max_value=300, value=120)
    blood_sugar = st.number_input("血糖（mmol/L）", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    blood_fat = st.number_input("血脂（mmol/L）", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

    # 在页面上显示用户输入的数据
    st.write("您的体温是:", body_temperature, "摄氏度")
    st.write("您的心率是:", heart_rate, "每分钟")
    st.write("您的血压是:", blood_pressure, "mmHg")
    st.write("您的血糖是:", blood_sugar, "mmol/L")
    st.write("您的血脂是:", blood_fat, "mmol/L")
    st.markdown("### 智能助手分析结果是:")
    # 将用户输入的数据保存在变量中，以便后续处理
    user_data = {
        '体温': body_temperature,
        '心率': heart_rate,
        '血压': blood_pressure,
        '血糖': blood_sugar,
        '血脂': blood_fat
    }
    if user_data:
        prompt_with_ai=prompt2.format(temperature=body_temperature,heart_rate=heart_rate,blood_pressure=blood_pressure,blood_sugar=blood_sugar,blood_fat=blood_fat)
        query = f"分析体温、心率、血压、血糖和血脂。体温={user_data['体温']}，心率={user_data['心率']}，血压={user_data['血压']}，血糖={user_data['血糖']}，血脂={user_data['血脂']}"
    
    # 使用本地向量数据库 qa2 运行查询
        analysis_result = qa2.run(query)
    
    # 显示分析结果
        st.write("分析结果：", analysis_result)   

    # 在这里可以进行后续处理，比如数据分析、可视化等

def page_query():
    st.markdown("## 问题咨询")
    option_type=st.selectbox(
        '你想要咨询什么方面的问题?',
    ('健康建议','医疗信息','紧急救助','心理健康')
    )
    type_input=get_text()
    st.markdown("### 可能的建议是: ")
    if type_input:
        prompt_with_ai=prompt1.format(type=option_type)
        query=type_input
        formatted_email=llm(prompt_with_ai)
        st.write(qa1.run(query))
def get_text():
    input_text=st.text_area(label="",placeholder="你的问题...",key="question_input")
    return input_text
def page_facial():
    st.title("舌部诊断")
    st.markdown("本功能的实现是通过Unet网络训练提取出舌头的语义类别，再通过支持向量机的方法训练舌部诊断模型，模型返回舌苔以及舌质的颜色。最终将获取的结果输入LangChain模型实现舌部诊断。")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1,col2=st.columns(2)
        with(col1):
            st.image(image, caption="您上传的图像", use_column_width=True)
        model = load_model('./utils/model_2_1.model')
        st.markdown("## 识别结果")
        seg_img=inference2(model,image)
        with(col2):
            st.image(seg_img,caption="语义分割后的图像",use_column_width=True)
        body,coat=detact_run.analysis(seg_img)
        # 使用OpenAI的GPT-3获取舌质和舌苔对应的症状和建议
        combined_color_info = f"舌质颜色 {body} 舌苔颜色 {coat}"
        combined_symptoms = get_symptoms_from_openai(combined_color_info)
        combined_suggestions = get_suggestions_from_openai(combined_color_info)
        # 显示结果
        st.write(f"您的舌质颜色是 {body}",f"您的舌苔颜色是 {coat}","以下是可能出现的症状以及建议")
        st.write(f"可能出现的症状: {combined_symptoms}")
        st.write(f"给您的建议是: {combined_suggestions}")    
        # 显示分割结果
        #st.image(segmentation_result, caption="识别结果", use_column_width=True)

def get_symptoms_from_openai(color_info):
    # 调用OpenAI的GPT-3或其他语言模型，获取舌质和舌苔颜色联合对应的症状
    os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
    openai.api_key =OPENAI_API_KEY
    #openai.api_key = os.getenv(OPENAI_API_KEY)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{color_info} 症状是什么？",
        max_tokens=100
    )
    return response.choices[0].text

def get_suggestions_from_openai(color_info):
    # 调用OpenAI的GPT-3或其他语言模型，获取舌质和舌苔颜色联合对应的建议
    os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
    openai.api_key =OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{color_info} 有什么建议？",
        max_tokens=100
    )
    return response.choices[0].text
def inference2(model,image):
    #img = load_img(image, color_mode="grayscale")
    image= image.resize((128,128))
    img_gray = image.convert ('L')
    x_1 = img_to_array(img_gray)
    x = np.expand_dims(x_1, axis=0)
    #plt.figure()
    #plt.imshow(np.uint8(x_1.reshape(128, 128)), cmap='gray', vmin=0, vmax=255)
    y = model.predict(x)
  #plt.figure()
  #plt.imshow(y.reshape((128, 128)))
  #y_overall = np.multiply(x_1, y.reshape(128, 128, 1))
  #plt.figure()
  #plt.imshow(np.uint8(y_overall.reshape(128, 128)), cmap='gray', vmin=0, vmax=255)
    #img = load_img(image, target_size=(128, 128))
    #x_1 = image.img_to_array(img)
    x_1=img_to_array(image)
    x = np.expand_dims(x_1, axis=0)
    y_overall = np.multiply(x, y.reshape(128, 128, 1))
  #plt.figure()
  #plt.imshow(np.uint8(y_overall.reshape(128, 128, 3)))
    yout=Image.fromarray(np.uint8(y_overall.reshape(128, 128, 3)))    
    return yout
    # 在About页面中显示数据和功能组件
session_state = st.session_state
if 'page' not in session_state:
    session_state['page'] = 'Home'

    # 导航栏
page = st.sidebar.radio('导航栏', ['健康状态分析', '问题咨询','舌部诊断'])

if page == '健康状态分析':
    page_home()
elif page == '问题咨询':
    page_query()
elif page == '舌部诊断':
    page_facial()

