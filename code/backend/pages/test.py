import streamlit as st
import sys
from os import path
import logging
import urllib.parse
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
from openai import AzureOpenAI
from batch.utilities.helpers.azure_document_intelligence_helper import AzureDocumentIntelligenceClient
from batch.utilities.helpers.env_helper import EnvHelper
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
import requests
from langchain.text_splitter import MarkdownHeaderTextSplitter, TokenTextSplitter, TextSplitter

sys.path.append(path.join(path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Test",
    page_icon=path.join("images", "favicon.ico"),
    layout="wide",
    menu_items=None,
)
mod_page_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(mod_page_style, unsafe_allow_html=True)




##############
# 测试上传文件含有空格和中文时，使用
# file_name = st.session_state.get("filename", "")
# url = urllib.parse.quote(file_name)
# url_plus = urllib.parse.quote_plus(file_name)


# st.write("file name:", file_name)
# st.write("quote url:", url)
# st.write("quote_plus url:", url_plus)
# st.write("original == quote?:", file_name == url)
# st.write("quote == quote_plus?:", url == url_plus)
##############目前将文档添加到search index时，filename还是会转义，和blob name不一致！





#################
# 测试 GPT 4v从环境变量获得max_tokens时，必须是强制转换为int类型
# image_path = "images\Microsoft_Fabric_Components_cropped_image_0.png"
# caption = ""
# api_base = env_helper.AZURE_OPENAI_ENDPOINT_4V
# api_key = env_helper.AZURE_OPENAI_API_KEY_4V
# deployment_name = env_helper.AZURE_OPENAI_DEPLOYMENT_4V
# api_version = env_helper.AZURE_OPENAI_API_VERSION_4V
# max_tokens = env_helper.MAX_TOKENS_4V

# gpt_4v_client = AzureOpenAI(
#     api_key=api_key,
#     api_version=api_version,
#     base_url=f"{api_base}/openai/deployments/{deployment_name}"
# )

# data_url = AzureDocumentIntelligenceClient.local_image_to_data_url(image_path)

# # We send both image caption and the image body to GPTv for better understanding
# text_message = f"Describe this image (note: it has image caption: {caption}):" if caption != "" else "Describe this image:"

# st.write("text_message:", text_message)

# response = gpt_4v_client.chat.completions.create(
#     model=deployment_name,
#     messages=[
#         { "role": "system", "content": "You are a helpful assistant." },
#         { "role": "user", "content": [  
#             { 
#                 "type": "text", 
#                 "text": text_message
#             },
#             { 
#                 "type": "image_url",
#                 "image_url": {
#                     "url": data_url
#                 }
#             }
#         ] } 
#     ],
#     max_tokens=max_tokens
# )

# img_description = response.choices[0].message.content
# st.write("img_description:", img_description)
################# done






#################################
# from PIL import Image
# import io
# import base64

# @staticmethod
# def image_to_data_url(image):
#     """
#     Converts a PIL Image to a data URL.

#     :param image: A PIL Image.
#     :return: A data URL representing the image.
#     """
#     # Convert the PIL Image to bytes
#     image_bytes = io.BytesIO()
#     image.save(image_bytes, format='PNG')
#     image_data = image_bytes.getvalue()

#     # Encode the image data
#     base64_encoded_data = base64.b64encode(image_data).decode('utf-8')

#     # Construct the data URL
#     return f"data:image/png;base64,{base64_encoded_data}"





# #################################
# st.write("测试pdfkit的使用")

# import pdfkit
# options = {
#     'encoding': "UTF-8",
#     'footer-center': "[page] of [topage]"
# }

# # url = "https://blog.csdn.net/weixin_44807854/article/details/110729371"
# # url = "https://tjj.beijing.gov.cn/tjsj_31433/sjjd_31444/202403/t20240319_3594001.html"
# # url = "https://new.qq.com/rain/a/20240321A05YQS00"
# url = "https://baike.baidu.com/item/%E4%BA%86%E5%87%A1%E5%9B%9B%E8%AE%AD/19450400"

# pdfkit.from_url(url, 'out.pdf', verbose=True)
# # pdf = pdfkit.from_url(url, options=options, verbose=True)
# # st.write("pdf: \n", pdf)

# st.write("out.pdf 已输出到当前目录中。")
# #################################





# #################################
# st.write("测试 WebBaseLoader 和 BeautifulSoup ")
# from typing import List
# from langchain.docstore.document import Document
# from langchain_community.document_loaders import WebBaseLoader
# import requests
# from bs4 import BeautifulSoup

# # url = "https://blog.csdn.net/weixin_44807854/article/details/110729371"
# url = "https://tjj.beijing.gov.cn/tjsj_31433/sjjd_31444/202403/t20240319_3594001.html"

# default_header_template = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
#     ";q=0.8",
#     "Accept-Language": "en-US,en;q=0.5",
#     "Referer": "https://www.google.com/",
#     "DNT": "1",
#     "Connection": "keep-alive",
#     "Upgrade-Insecure-Requests": "1",
# }

# session = requests.Session()
# header_template = default_header_template.copy()
# if not header_template.get("User-Agent"):
#     try:
#         from fake_useragent import UserAgent
#         header_template["User-Agent"] = UserAgent().random
#     except ImportError:
#         logger.info(
#             "fake_useragent not found, using default user agent."
#             "To get a realistic header for requests, "
#             "`pip install fake_useragent`."
#         )
# session.headers = dict(header_template)
# html_doc = session.get(url)
# if url.endswith(".xml"):
#     parser = "xml"
# else:
#     parser = "html.parser"

# soup = BeautifulSoup(html_doc.text, parser)

# # documents: List[Document] = WebBaseLoader(url).load()
# # st.write("documents:", documents)

# my_url_content = WebBaseLoader(url).scrape()

# st.write("html_doc type:", type(html_doc))
# st.write("html_doc:", html_doc)
# st.write("soup type:", type(soup))
# st.write("soup:", soup)
# st.write("soup title:", soup.title)


# st.write("测试 WebBaseLoader 和 BeautifulSoup ")
#################################





#################################
#  https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
#  测试使用st.data_editor获取用户选择的行
# import pandas as pd

# df = pd.DataFrame(
#     {
#         "Animal": ["Lion", "Elephant", "Giraffe", "Monkey", "Zebra"],
#         "Habitat": ["Savanna", "Forest", "Savanna", "Forest", "Savanna"],
#         "Lifespan (years)": [15, 60, 25, 20, 25],
#         "Average weight (kg)": [190, 5000, 800, 10, 350],
#     }
# )

# def dataframe_with_selections(df):
#     df_with_selections = df.copy()
#     df_with_selections.insert(0, "Select", False)

#     # Get dataframe row-selections from user with st.data_editor
#     edited_df = st.data_editor(
#         df_with_selections,
#         hide_index=True,
#         column_config={"Select": st.column_config.CheckboxColumn(required=True)},
#         disabled=df.columns,
#     )

#     # Filter the dataframe using the temporary column, then drop the column
#     selected_rows = edited_df[edited_df.Select]
#     return selected_rows.drop('Select', axis=1)


# selection = dataframe_with_selections(df)
# st.write("Your selection:")
# st.write(selection)

# df_with_selections = df.copy()
# st.write("df_with_selections:", df_with_selections)

# df_with_selections.insert(0, "Select", False)
# st.write("df_with_selections:", df_with_selections)

# st.write("df.columns:", df.columns)
# edited_df = st.data_editor(
#     df_with_selections,
#     hide_index=True,
#     column_config={"Select": st.column_config.CheckboxColumn(required=True)},
#     disabled=df.columns,
# )
# st.write("edited_df:", edited_df)

# st.write("edited_df.Select:", edited_df.Select)
# selected_rows = edited_df[edited_df.Select]
# st.write("selected_rows:", selected_rows)

# selection = selected_rows.drop('Select', axis=1)
# st.write("Your selection:", selection)

#################################





#################################
#测试将 MD文件进行分割

# Split the document into chunks base on markdown headers.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),  
    ("#######", "Header 7"), 
    ("########", "Header 8")
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_text = """

"""

splits = text_splitter.split_text(markdown_text)
for text in splits:
    st.write("#" * 10)
    st.markdown(text.page_content)


#################################


# blob_client = AzureBlobStorageClient()
# files=blob_client.get_all_files()
# st.write("blob list all files:", files)


#################################
st.write("#" * 60)
st.write("process document: DI处理完了, 改写到 document loading and chunking")
st.write("process document: 处理 URL.  convert to pdf and upload to blob storage, and then 06.Process?")
st.write("process document: 处理 text")
st.write("process document: 处理 word DOCX")
st.write("测试上传 0KB的文件报错")
#################################

