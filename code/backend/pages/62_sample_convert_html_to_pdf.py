# coding: utf-8

# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_convert_html_to_pdf.py
DESCRIPTION:
    This sample demonstrates how to convert a html file to the pdf file.
PREREQUISITES:
    Before using this function, need to install following required component:
        1).Install python-pdfkit:
            "$ pip install pdfkit"
        2).Install wkhtmltopdf:
            -Debian/Ubuntu:
                "$ sudo apt-get install wkhtmltopdf"
            -macOS:
                "$ brew install homebrew/cask/wkhtmltopdf"
            -Windows and other options: check https://wkhtmltopdf.org/downloads.html for wkhtmltopdf binary installers
    More information about pdfkit, reference from https://pypi.org/project/pdfkit/.
USAGE:
    python sample_convert_html_to_pdf.py
"""


import streamlit as st
import pdfkit
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient


env_helper: EnvHelper = EnvHelper()
blob_client = AzureBlobStorageClient()

def add_urls():
    urls = st.session_state.urls.split("\n")

    options = {
        'encoding': "UTF-8",
    }

    i = 1
    for url in urls:
        # pdfkit.from_url(url, f'out{i}.pdf', options=options)
        pdf = pdfkit.from_url(url, options=options)
        blob_client.upload_file(pdf, f"out{i}.pdf", metadata={"title": f"out{i}.pdf"})
        # st.write("pdf type:", type(pdf))
        # st.write("pdf:", pdf)
        i += 1


with st.expander("Add URLs", expanded=True):
    st.text_area(
        "Add URLs, convert to PDF and upload to Azure Blob Storage",
        placeholder="PLACE YOUR URLS HERE SEPARATED BY A NEW LINE",
        height=100,
        key="urls",
    )

    st.button(
        "Process web pages",
        on_click=add_urls,
        key="add_url",
    )


# https://tjj.beijing.gov.cn/tjsj_31433/sjjd_31444/202403/t20240319_3594001.html
# https://new.qq.com/rain/a/20240321A05YQS00
# https://baike.baidu.com/item/%E4%BA%86%E5%87%A1%E5%9B%9B%E8%AE%AD/19450400
# https://blog.csdn.net/weixin_44807854/article/details/110729371