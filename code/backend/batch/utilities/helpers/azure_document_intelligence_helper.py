from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeOutputOption, AnalyzeResult
import re
from .env_helper import EnvHelper
from .llm_helper import LLMHelper
from .azure_blob_storage_client import AzureBlobStorageClient
import traceback
import logging



logger = logging.getLogger(__name__)

class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        self.env_helper: EnvHelper = EnvHelper()
        self.llm_helper: LLMHelper = LLMHelper()
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

    def gen_figure_description(self, figure_url, caption, max_tokens=1000):
        """
        Generates a description for an image using the GPT vision model.

        Parameters:
        - figure_url (str): The url to the image file.
        - caption (str): The caption for the image.

        Returns:
        - figure_description (str): The generated description for the image.
        """

        gpt_vision_client = self.llm_helper.openai_client

        # We send both image caption and the image body to GPT for better understanding
        if caption:
            text_message = f"Describe the image and generate the description text using the language of the text elements in the image(note: it has image caption: {caption}):"
        else:
            text_message = "Describe the image and generate the description text using the language of the text elements in the image:"

        response = gpt_vision_client.chat.completions.create(
            model=self.env_helper.AZURE_OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
- You are an assistant that generates rich and detailed descriptions of images.
- When extracting information from images, be accurate and comprehensive, including non-textual elements and numbers. Pay attention to accurately capturing the positive and negative signs of numbers.
- For images primarily containing text and numbers, use OCR to accurately extract and transcribe the text and numbers from the image. Utilize the position of the text and charts to enhance the overall description.
- For images containing visual content such as bar charts, line charts, pie charts, scatter plots, and stacked charts, first extract the text and numbers, then analyze and describe the trends, patterns, and insights conveyed by these visualizations.
- Pay attention to the correspondence between numbers, axes, and legends (including colors and shapes).
- Ensure to interpret and explain the significance of the data trends and patterns in the context of the image.
- Automatically detect the primary language used in the image text and generate the description in that language to maintain consistency. For example, if the text in the image is in Chinese, generate the description in Chinese.
- Avoid abbreviations and maintain complete sentences to provide a thorough explanation of the image content.
- Ensure that the description captures the context and purpose of the image, providing insights into any relevant information implied by the visual elements.
- Format the generated description using Markdown to ensure clear and structured presentation of information. But DO NOT use Markdown header symbol #.
"""
                },
                { "role": "user", "content": [
                    {
                        "type": "text",
                        "text": text_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": figure_url
                        }
                    }
                ] }
            ],
            max_tokens=max_tokens,
            temperature = 0,
            stream=False
        )

        figure_description = response.choices[0].message.content
        return figure_description

    # Update markdown figure content section with the description from GPT vision model
    def update_figure_description(self, md_content, figure_description, idx):
        """
        Updates the figure description in the Markdown content.

        Args:
            md_content (str): The original Markdown content.
            figure_description (str): The new description for the image.
            idx (int): The index of the figure.

        Returns:
            str: The updated Markdown content with the new figure description.
        """

        figure_list = []
        start_pos = 0
        while True:
            start_tag = md_content.find("<figure>", start_pos)
            if start_tag == -1:
                break
            end_tag = md_content.find("</figure>", start_tag)
            if end_tag == -1:
                break
            figure_list.append((start_tag, end_tag + len("</figure>")))
            start_pos = end_tag + len("</figure>")

        if idx < len(figure_list):
            start, end = figure_list[idx]
            old_figure_block = md_content[start:end]

            # Find <figcaption>...</figcaption> if present
            cap_start = old_figure_block.find("<figcaption>")
            if cap_start != -1:
                cap_end = old_figure_block.find("</figcaption>", cap_start)
                if cap_end != -1:
                    figcaption_block = old_figure_block[cap_start : cap_end + len("</figcaption>")]
                else:
                    figcaption_block = ""
            else:
                figcaption_block = ""

            new_block = f"<figure>\n{figcaption_block}\n{figure_description}\n</figure>"
            return md_content[:start] + new_block + md_content[end:]

        return md_content


    # Analyze a document with Azure AI Document Intelligence Layout model and update figure description in the markdown output
    def analyze_layout(self, input_file_url:str, input_file_name:str, gen_figure_desc:bool=True):
        """
        Analyzes the layout of a potentially large document in 100-page chunks
        and extracts figures along with their descriptions, then updates the
        markdown output with the new descriptions.

        Args:
            input_file_url (str): The blob url of the input file.
            input_file_name (str): The blob name of the input file.
            gen_figure_desc (bool): if True then extract figures and generate description, if False then do not process figures.

        Returns:
            str: The updated Markdown content with the new figure description.
        """
        try:
            chunk_size = 100
            start_page = 1
            all_md_content = ""

            while True:
                end_page = start_page + chunk_size - 1
                if end_page < start_page:
                    break

                pages_param = f"{start_page}-{end_page}"
                poller = self.document_intelligence_client.begin_analyze_document(
                    "prebuilt-layout",
                    body=AnalyzeDocumentRequest(url_source=input_file_url),
                    pages=pages_param,
                    output_content_format=DocumentContentFormat.MARKDOWN,
                    output=[AnalyzeOutputOption.FIGURES],
                )
                result: AnalyzeResult = poller.result()
                if not result.pages:
                    print(f"No pages found for file {input_file_name} in range {pages_param}.")
                    break

                model_id = result.model_id
                chunk_md_content = result.content
                operation_id = poller.details["operation_id"]

                # Extract figures and generate descriptions
                if gen_figure_desc and result.figures:
                    for idx, figure in enumerate(result.figures):
                        response = self.document_intelligence_client.get_analyze_result_figure(
                            model_id=model_id, result_id=operation_id, figure_id=figure.id
                        )
                        figure_caption = figure.caption.content if figure.caption else ""
                        figure_filename = f"converted/{input_file_name}-{figure.id}.png"
                        figure_file_url = self.blob_client.upload_file(bytes_data=response, file_name=figure_filename)
                        figure_description = self.gen_figure_description(figure_file_url, figure_caption, int(self.env_helper.AZURE_OPENAI_MAX_TOKENS))
                        chunk_md_content = self.update_figure_description(chunk_md_content, figure_description, idx)
                elif result.figures:
                    print(f"Figures found in file {input_file_name} for pages {pages_param}, but set it to not generate figure description.")
                else:
                    print(f"No figures found in file {input_file_name} for pages {pages_param}.")

                all_md_content += chunk_md_content

                if len(result.pages) < chunk_size:
                    break
                start_page += chunk_size

            converted_file_name = f"converted/{input_file_name}_converted.md"
            converted_file_url = self.blob_client.upload_file(all_md_content.encode('utf-8'), converted_file_name)
            self.blob_client.upsert_blob_metadata(input_file_name, {"converted": "true"})

            return all_md_content
        except Exception as e:
            logger.error(f"An error occurred while analyzing the document layout: {e}")
            raise ValueError(f"Error: {traceback.format_exc()}. Error: {e}")
