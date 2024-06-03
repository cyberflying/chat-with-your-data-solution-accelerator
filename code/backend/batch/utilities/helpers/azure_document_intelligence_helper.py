import os
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat
# Crop figure from the document (pdf or image) based on the bounding box
import requests
from PIL import Image
import io
import fitz  # PyMuPDF
import mimetypes
# Use Azure OpenAI (GPT-4V model) to understand the semantics of the figure content
from openai import AzureOpenAI
import base64
from .env_helper import EnvHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
from docx import Document
import traceback
import logging



logger = logging.getLogger(__name__)

class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        self.env_helper: EnvHelper = EnvHelper()
        self.blob_client = AzureBlobStorageClient()
        self.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str = (self.env_helper.AZURE_FORM_RECOGNIZER_ENDPOINT)
        if self.env_helper.AZURE_AUTH_TYPE == "rbac":
            self.document_intelligence_client  = DocumentIntelligenceClient(
                endpoint=self.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
        else:
            self.AZURE_FORM_RECOGNIZER_KEY: str = self.env_helper.AZURE_FORM_RECOGNIZER_KEY

            self.document_intelligence_client  = DocumentIntelligenceClient(
                endpoint=self.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(self.AZURE_FORM_RECOGNIZER_KEY),
            )



    # Crop figure from the document (pdf or image) based on the bounding box
    @staticmethod
    def crop_image_from_image(image_data, page_number, bounding_box):
        """
        Crops an image based on a bounding box.

        :param image_data: BytesIO object containing the image data.
        :param page_number: The page number of the image to crop (for TIFF format).
        :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
        :return: A cropped image.
        :rtype: PIL.Image.Image
        """
        with Image.open(image_data) as img:
            if img.format == "TIFF":
                # Open the TIFF image
                img.seek(page_number)
                img = img.copy()

            # The bounding box is expected to be in the format (left, upper, right, lower).
            cropped_image = img.crop(bounding_box)
            return cropped_image


    @staticmethod
    def crop_image_from_pdf_page(pdf_data, page_number, bounding_box):
        """
        Crops a region from a given page in a PDF and returns it as an image.

        :param pdf_data: BytesIO object containing the PDF data.
        :param page_number: The page number to crop from (0-indexed).
        :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
        :return: A PIL Image of the cropped area.
        """
        doc = fitz.open("pdf", pdf_data.read())
        page = doc.load_page(page_number)
        
        # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
        bbx = [x * 72 for x in bounding_box]
        rect = fitz.Rect(bbx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img


    @staticmethod
    def crop_image_from_docx(docx_data, image_number, bounding_box):
        """
        Crops an image from a docx file.

        :param docx_data: BytesIO object containing the docx data.
        :param image_number: The number of the image to crop (0-indexed).
        :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
        :return: A cropped image.
        :rtype: PIL.Image.Image
        """
        doc = Document(docx_data)
        images = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_data = rel.blob
                image = Image.open(io.BytesIO(image_data))
                images.append(image)

        if image_number < len(images):
            # The bounding box is expected to be in the format (left, upper, right, lower).
            cropped_image = images[image_number].crop(bounding_box)
            return cropped_image
        else:
            raise ValueError(f"Image number {image_number} not found in docx file")


    @staticmethod
    def crop_image_from_file(file_path_or_url, page_number, bounding_box):
        """
        Crop an image from a file.

        Args:
            file_path_or_url (str): The path or URL to the file.
            page_number (int): The page number (for PDF and TIFF files, 0-indexed).
            bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

        Returns:
            A PIL Image of the cropped area.
        """
        if os.path.exists(file_path_or_url):
            # It's a file path
            with open(file_path_or_url, 'rb') as f:
                file_data = io.BytesIO(f.read())
        else:
            # It's a URL
            response = requests.get(file_path_or_url)
            file_data = io.BytesIO(response.content)

        mime_type = mimetypes.guess_type(file_path_or_url.split('?')[0])[0]
        
        if mime_type == "application/pdf":
            return AzureDocumentIntelligenceClient.crop_image_from_pdf_page(file_data, page_number, bounding_box)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return AzureDocumentIntelligenceClient.crop_image_from_docx(file_data, page_number, bounding_box)
        else:
            return AzureDocumentIntelligenceClient.crop_image_from_image(file_data, page_number, bounding_box)

    @staticmethod
    def image_to_data_url(image_data):
        """
        Converts a PIL Image to a data URL.
    
        :param image: A PIL Image.
        :return: A data URL representing the image.
        """

        # Encode the image data
        base64_encoded_data = base64.b64encode(image_data).decode('utf-8')
    
        # Construct the data URL
        return f"data:image/png;base64,{base64_encoded_data}"



    # Use Azure OpenAI (GPT-4V model) to understand the semantics of the figure content
    # @staticmethod
    def gen_image_description_with_gpt(self, api_base, api_key:str, deployment_name, api_version, image_data, caption, max_tokens=2000):
        """
        Generates a description for an image using the GPT-4V model.

        Parameters:
        - api_base (str): The base URL of the API.
        - api_key (str): The API key for authentication.
        - deployment_name (str): The name of the deployment.
        - api_version (str): The version of the API.
        - image_path (str): The path to the image file.
        - caption (str): The caption for the image.

        Returns:
        - img_description (str): The generated description for the image.
        """
        gpt_4v_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=f"{api_base}/openai/deployments/{deployment_name}"
        )

        data_url = AzureDocumentIntelligenceClient.image_to_data_url(image_data)

        # We send both image caption and the image body to GPTv for better understanding
        if caption:
            text_message = f"Describe this image in simplified Chinese: (note: it has image caption: {caption}):"
        else:
            text_message = "Describe this image in simplified Chinese:"   #"Describe this image and use the same style and language as the text within the image:"
        response = gpt_4v_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system", 
                    "content": """You are an assistant that generates rich descriptions of images.
You need to be accurate in the information you extract and detailed in the descriptons you generate.
Do not abbreviate anything and do not shorten sentances. Explain the image completely.
If you are provided with an image of a flow chart, describe the flow chart in detail.
If the image is mostly text, use OCR to extract the text as it is displayed in the image.""" 
                },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": text_message
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ] } 
            ],
            max_tokens=max_tokens,
            stream=False
        )

        img_description = response.choices[0].message.content

        return img_description



    # Update markdown figure content section with the description from GPT-4V model
    def update_figure_description(self, md_content, img_description, idx):
        """
        Updates the figure description in the Markdown content.

        Args:
            md_content (str): The original Markdown content.
            img_description (str): The new description for the image.
            idx (int): The index of the figure.

        Returns:
            str: The updated Markdown content with the new figure description.
        """

        # The substring you're looking for
        start_substring = f"![](figures/{idx})"
        end_substring = "</figure>"
        new_string = f"<!-- FigureContent=\"{img_description}\" -->"
        
        new_md_content = md_content
        # Find the start and end indices of the part to replace
        start_index = md_content.find(start_substring)
        if start_index != -1:  # if start_substring is found
            start_index += len(start_substring)  # move the index to the end of start_substring
            end_index = md_content.find(end_substring, start_index)
            if end_index != -1:  # if end_substring is found
                # Replace the old string with the new string
                new_md_content = md_content[:start_index] + new_string + md_content[end_index:]
        
        return new_md_content



    # Analyze a document with Azure AI Document Intelligence Layout model and update figure description in the markdown output

    def analyze_layout(self, input_file_url:str, input_file_name:str, gen_image_description:bool=True):
        """
        Analyzes the layout of a document and extracts figures along with their descriptions, then update the markdown output with the new description.

        Args:
            input_file_path (str): The path to the input document file.
            output_folder (str): The path to the output folder where the cropped images will be saved.

        Returns:
            str: The updated Markdown content with figure descriptions.

        """

        # with open(input_file_path, "rb") as f:
        #     poller = self.document_intelligence_client.begin_analyze_document("prebuilt-layout", analyze_request=f, content_type="application/octet-stream", output_content_format=ContentFormat.MARKDOWN)

        # poller = self.document_intelligence_client.begin_analyze_document("prebuilt-layout", analyze_request=bytes_data, content_type="application/octet-stream", output_content_format=ContentFormat.MARKDOWN)

        try:
            analyze_request = AnalyzeDocumentRequest(url_source=input_file_url)
            poller = self.document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", 
                analyze_request=analyze_request, 
                content_type="application/json", 
                output_content_format=ContentFormat.MARKDOWN 
            )
            result = poller.result()
            md_content = result.content
            
            
            if result.figures:
                # print("1 Figures:")
                for idx, figure in enumerate(result.figures):
                    figure_content = ""
                    img_description = ""
                    # print(f"2 Figure #{idx} has the following spans: {figure.spans}")
                    for i, span in enumerate(figure.spans):
                        # print(f"3 Span #{i}: {span}")
                        figure_content += md_content[span.offset:span.offset + span.length]
                    # print(f"4 Original figure content in markdown: {figure_content}")

                    # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
                    if figure.caption:
                        caption_region = figure.caption.bounding_regions
                        # print(f"5 \tCaption: {figure.caption.content}")
                        # print(f"6 \tCaption bounding region: {caption_region}")
                        for region in figure.bounding_regions:
                            if region not in caption_region:
                                # print(f"7 \tFigure body bounding regions: {region}")
                                # To learn more about bounding regions, see https://aka.ms/bounding-region
                                boundingbox = (
                                        region.polygon[0],  # x0 (left)
                                        region.polygon[1],  # y0 (top)
                                        region.polygon[4],  # x1 (right)
                                        region.polygon[5]   # y1 (bottom)
                                    )
                                # print(f"8 \tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")
                                cropped_image = AzureDocumentIntelligenceClient.crop_image_from_file(input_file_url, region.page_number - 1, boundingbox) # page_number is 1-indexed
                                image_bytes_io = io.BytesIO()
                                cropped_image.save(image_bytes_io, format='PNG')
                                image_data = image_bytes_io.getvalue()
                                output_file = f"converted/{input_file_name}_cropped_image_{idx}.png"
                                cropped_image_file_url = self.blob_client.upload_file(bytes_data=image_data, file_name=output_file)
                                # print(f"9 \tFigure {idx} cropped and saved as {cropped_image_file_url}")

                                if gen_image_description:
                                    img_description += self.gen_image_description_with_gpt(self.env_helper.AZURE_OPENAI_ENDPOINT, self.env_helper.AZURE_OPENAI_API_KEY, self.env_helper.AZURE_OPENAI_MODEL, self.env_helper.AZURE_OPENAI_API_VERSION, image_data, figure.caption.content, int(self.env_helper.AZURE_OPENAI_MAX_TOKENS))
                                    # print(f"10 \tDescription of figure {idx}: {img_description}")
                    else:
                        # print("11 \tNo caption found for this figure.")
                        for region in figure.bounding_regions:
                            # print(f"12 \tFigure body bounding regions: {region}")
                            # To learn more about bounding regions, see https://aka.ms/bounding-region
                            boundingbox = (
                                    region.polygon[0],  # x0 (left)
                                    region.polygon[1],  # y0 (top
                                    region.polygon[4],  # x1 (right)
                                    region.polygon[5]   # y1 (bottom)
                                )
                            # print(f"13 \tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")

                            cropped_image = AzureDocumentIntelligenceClient.crop_image_from_file(input_file_url, region.page_number - 1, boundingbox) # page_number is 1-indexed
                            image_bytes_io = io.BytesIO()
                            cropped_image.save(image_bytes_io, format='PNG')
                            image_data = image_bytes_io.getvalue()
                            output_file = f"converted/{input_file_name}_cropped_image_{idx}.png"
                            cropped_image_file_url = self.blob_client.upload_file(bytes_data=image_data, file_name=output_file)
                            # print(f"14 \tFigure {idx} cropped and saved as {cropped_image_file_url}")

                            if gen_image_description:
                                img_description += self.gen_image_description_with_gpt(self.env_helper.AZURE_OPENAI_ENDPOINT, self.env_helper.AZURE_OPENAI_API_KEY, self.env_helper.AZURE_OPENAI_MODEL, self.env_helper.AZURE_OPENAI_API_VERSION, image_data, "", int(self.env_helper.AZURE_OPENAI_MAX_TOKENS))
                                # print(f"15 \tDescription of figure {idx}: {img_description}")
                    
                    # replace_figure_description(figure_content, img_description, idx)
                    md_content = self.update_figure_description(md_content, img_description, idx)
                    figure_file_name = output_file.split("/")[-1].replace(" ", "%20")
                    md_content = md_content.replace(f"figures/{idx}", f"{figure_file_name}")

            converted_file_name = f"converted/{input_file_name}_converted.md"
            converted_file_url = self.blob_client.upload_file(md_content.encode('utf-8'), converted_file_name)
            self.blob_client.upsert_blob_metadata(input_file_name, {"converted": "true"})

            return md_content
        except Exception as e:
            logger.error(f"An error occurred while analyzing the document layout: {e}") 
            raise ValueError(f"Error: {traceback.format_exc()}. Error: {e}")