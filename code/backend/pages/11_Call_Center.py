import json
import time
import re
import uuid
import streamlit as st
import streamlit_extras.stateful_button as stx
from streamlit_js_eval import streamlit_js_eval
from scipy.io import wavfile
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction, AbstractiveSummaryAction
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient
import sys
from os import path
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.llm_helper import LLMHelper
import pandas as pd



# page layout configuration
sys.path.append(path.join(path.dirname(__file__), ".."))
st.set_page_config(
    page_title="Call Center",
    page_icon=path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load the common CSS
load_css("pages/common.css")


env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

speech_resource_id = f"/subscriptions/{env_helper.AZURE_SUBSCRIPTION_ID}/resourceGroups/{env_helper.AZURE_RESOURCE_GROUP}/providers/Microsoft.CognitiveServices/accounts/{env_helper.AZURE_SPEECH_SERVICE_NAME}"
speech_region = env_helper.AZURE_SPEECH_SERVICE_REGION
speech_authorizationToken = "aad#" + speech_resource_id + "#" + token.token

language_endpoint = env_helper.AZURE_LANGUAGE_ENDPOINT

cosmos_endpoint = f"https://{env_helper.AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/"
cosmos_database_name = env_helper.AZURE_COSMOSDB_DATABASE
cosmos_container_name = "CallTranscripts"


@st.cache_data
def create_transcription_request(audio_file, speech_recognition_language="en-US"):
    """Transcribe the contents of an audio file. Key assumptions:
    - The audio file is in WAV format.
    - The audio file is mono.
    - The audio file has a sample rate of 16 kHz."""

    # Create an instance of a speech config.
    speech_config = speechsdk.SpeechConfig(auth_token=speech_authorizationToken, region=speech_region)
    speech_config.speech_recognition_language=speech_recognition_language

    # Prepare audio settings for the wave stream
    channels = 1
    bits_per_sample = 16
    samples_per_second = 16000

    # Create audio configuration using the push stream
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config, audio_config)
    all_results = []

    def handle_final_result(evt):
        all_results.append(evt.result.text)

    done = False

    def stop_cb(evt):
        print(f'CLOSING on {evt}')
        nonlocal done
        done= True

    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print(f'SESSION STARTED: {evt}'))
    transcriber.session_stopped.connect(lambda evt: print(f'SESSION STOPPED {evt}'))
    transcriber.canceled.connect(lambda evt: print(f'CANCELED {evt}'))
    # stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)
    transcriber.start_transcribing_async()
    # Read the whole wave files at once and stream it to sdk
    _, wav_data = wavfile.read(audio_file)
    stream.write(wav_data.tobytes())
    stream.close()
    while not done:
        time.sleep(.5)
    transcriber.stop_transcribing_async()

    return all_results


def create_live_transcription_request(speech_recognition_language="en-US"):
    # Create an instance of a speech config.
    speech_config = speechsdk.SpeechConfig(auth_token=speech_authorizationToken, region=speech_region)
    speech_config.speech_recognition_language=speech_recognition_language
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config)

    done = False

    def handle_final_result(evt):
        all_results.append(evt.result.text)
        # print(evt.result.text)
        st.write(evt.result.text)

    all_results = []

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous transcription upon receiving an event `evt`"""
        # print('CLOSING {}'.format(evt))
        st.write('CLOSING {}'.format(evt))
        nonlocal done
        done = True

    # Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    transcriber.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    transcriber.canceled.connect(lambda evt: print('CANCELLED {}'.format(evt)))
    # stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)

    transcriber.start_transcribing_async()

    # Streamlit refreshes the page on each interaction,
    # so a clean start and stop isn't really possible with button presses.
    # Instead, we're constantly updating transcription results, so that way,
    # when the user clicks the button to stop, we can just stop updating the results.
    # This might not capture the final message, however, if the user stops before
    # we receive the message--we won't be able to call the stop event.
    while not done:
        st.session_state.transcription_results = all_results
        time.sleep(1)

    return


def make_azure_openai_chat_request(system, call_contents):
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": call_contents}
    ]
    return llm_helper.get_chat_completion(messages=messages)


@st.cache_data
def is_call_in_compliance(call_contents, include_recording_message, is_relevant_to_topic):
    """Analyze a call for relevance and compliance."""
    joined_call_contents = ' '.join(call_contents)
    if include_recording_message:
        include_recording_message_text = "2. Was the caller aware that the call was being recorded?"
    else:
        include_recording_message_text = ""
    if is_relevant_to_topic:
        is_relevant_to_topic_text = "3. Was the call relevant to the hotel and resort industry?"
    else:
        is_relevant_to_topic_text = ""
    system = f"""
        You are an automated analysis system for Contoso Suites.
        Contoso Suites is a luxury hotel and resort chain with locations
        in a variety of Caribbean nations and territories.
        You are analyzing a call for relevance and compliance.
        You will only answer the following questions based on the call contents:
        1. Was there vulgarity on the call?
        {include_recording_message_text}
        {is_relevant_to_topic_text}
    """
    response = make_azure_openai_chat_request(system, joined_call_contents)
    return response.choices[0].message.content


@st.cache_data
def generate_extractive_summary(call_contents, contents_language="en-us"):
    """Generate an extractive summary of a call transcript."""

    # The call_contents parameter is formatted as a list of strings.
    # Join them together with spaces to pass in as a single document.
    joined_call_contents = ' '.join(call_contents)

    client = TextAnalyticsClient(language_endpoint, credential)
    # Call the begin_analyze_actions method on your client, passing in the joined
    # call_contents as an array and an ExtractiveSummaryAction with a max_sentence_count of 2.
    poller = client.begin_analyze_actions(
        [joined_call_contents],
        language = contents_language,
        actions = [
            ExtractiveSummaryAction(max_sentence_count=2)
        ]
    )

    # Extract the summary sentences and merge them into a single summary string.
    for result in poller.result():
        summary_result = result[0]
        if summary_result.is_error:
            st.error(f'Extractive summary resulted in an error with code "{summary_result.code}" and message "{summary_result.message}"')
            return ''

        extractive_summary = " ".join([sentence.text for sentence in summary_result.sentences])

    # Return the summary as a JSON object in the shape '{"call-summary":extractive_summary}'
    return json.loads('{"call-summary":"' + extractive_summary + '"}')


@st.cache_data
def generate_abstractive_summary(call_contents, contents_language="en-us"):
    """Generate an abstractive summary of a call transcript."""

    # The call_contents parameter is formatted as a list of strings.
    # Join them together with spaces to pass in as a single document.
    joined_call_contents = ' '.join(call_contents)

    client = TextAnalyticsClient(language_endpoint, credential)

    # Call the begin_analyze_actions method on your client,
    # passing in the joined call_contents as an array
    # and an AbstractiveSummaryAction with a sentence_count of 2.
    poller = client.begin_analyze_actions(
        [joined_call_contents],
        language = contents_language,
        actions = [
            AbstractiveSummaryAction(sentence_count=2)
        ]
    )

    # Extract the summary sentences and merge them into a single summary string.
    for result in poller.result():
        summary_result = result[0]
        if summary_result.is_error:
            st.error(f'...Is an error with code "{summary_result.code}" and message "{summary_result.message}"')
            return ''

        abstractive_summary = " ".join([summary.text for summary in summary_result.summaries])

    # Return the summary as a JSON object in the shape '{"call-summary":abstractive_summary}'
    return json.loads('{"call-summary":"' + abstractive_summary + '"}')


@st.cache_data
def generate_query_based_summary(call_contents):
    """Generate a query-based summary of a call transcript."""

    # The call_contents parameter is formatted as a list of strings.
    # Join them together with spaces to pass in as a single document.
    joined_call_contents = ' '.join(call_contents)

    # Write a system prompt that instructs the large language model to:
    system = """
- Write a five-word summary and label it as call-title.
- Write a two-sentence summary and label it as call-summary.
- Output the results in JSON format.
    """
    st.write(f"**system prompt**: {system}")
    response = make_azure_openai_chat_request(system, joined_call_contents)

    # Return the summary.
    return response.choices[0].message.content


@st.cache_data
def create_sentiment_analysis_and_opinion_mining_request(call_contents):
    """Analyze the sentiment of a call transcript and mine opinions. Key assumptions:
    - Azure AI Services Language service endpoint and key stored in Streamlit secrets."""

    # The call_contents parameter is formatted as a list of strings.
    # Join them together with spaces to pass in as a single document.
    joined_call_contents = ' '.join(call_contents)

    client = TextAnalyticsClient(language_endpoint, credential)

    # Analyze sentiment of call transcript, enabling opinion mining.
    result = client.analyze_sentiment([joined_call_contents], show_opinion_mining=True)

    # Retrieve all document results that are not an error.
    doc_result = [doc for doc in result if not doc.is_error]

    # The output format is a JSON document with the shape:
    # {
    #     "sentiment": document_sentiment,
    #     "sentiment-scores": {
    #         "positive": document_positive_score_as_two_decimal_float,
    #         "neutral": document_neutral_score_as_two_decimal_float,
    #         "negative": document_negative_score_as_two_decimal_float
    #     },
    #     "sentences": [
    #         {
    #             "text": sentence_text,
    #             "sentiment": document_sentiment,
    #             "sentiment-scores": {
    #                 "positive": document_positive_score_as_two_decimal_float,
    #                 "neutral": document_neutral_score_as_two_decimal_float,
    #                 "negative": document_negative_score_as_two_decimal_float
    #             },
    #             "mined_opinions": [
    #                 {
    #                     "target-sentiment": opinion_sentiment,
    #                     "target-text": opinion_target,
    #                     "target-scores": {
    #                         "positive": document_positive_score_as_two_decimal_float,
    #                         "neutral": document_neutral_score_as_two_decimal_float,
    #                         "negative": document_negative_score_as_two_decimal_float
    #                     },
    #                     "assessments": [
    #                       {
    #                         "assessment-sentiment": assessment_sentiment,
    #                         "assessment-text": assessment_text,
    #                         "assessment-scores": {
    #                             "positive": document_positive_score_as_two_decimal_float,
    #                             "negative": document_negative_score_as_two_decimal_float
    #                         }
    #                       }
    #                     ]
    #                 }
    #             ]
    #         }
    #     ]
    # }
    sentiment = {}

    # Assign the correct values to the JSON object.
    for document in doc_result:
        sentiment["sentiment"] = document.sentiment
        sentiment["sentiment-scores"] = {
            "positive": document.confidence_scores.positive,
            "neutral": document.confidence_scores.neutral,
            "negative": document.confidence_scores.negative
        }

        sentences = []
        for s in document.sentences:
            sentence = {}
            sentence["text"] = s.text
            sentence["sentiment"] = s.sentiment
            sentence["sentiment-scores"] = {
                "positive": s.confidence_scores.positive,
                "neutral": s.confidence_scores.neutral,
                "negative": s.confidence_scores.negative
            }

            mined_opinions = []
            for mined_opinion in s.mined_opinions:
                opinion = {}
                opinion["target-text"] = mined_opinion.target.text
                opinion["target-sentiment"] = mined_opinion.target.sentiment
                opinion["sentiment-scores"] = {
                    "positive": mined_opinion.target.confidence_scores.positive,
                    "negative": mined_opinion.target.confidence_scores.negative,
                }

                opinion_assessments = []
                for assessment in mined_opinion.assessments:
                    opinion_assessment = {}
                    opinion_assessment["text"] = assessment.text
                    opinion_assessment["sentiment"] = assessment.sentiment
                    opinion_assessment["sentiment-scores"] = {
                        "positive": assessment.confidence_scores.positive,
                        "negative": assessment.confidence_scores.negative
                    }
                    opinion_assessments.append(opinion_assessment)

                opinion["assessments"] = opinion_assessments
                mined_opinions.append(opinion)

            sentence["mined_opinions"] = mined_opinions
            sentences.append(sentence)

        sentiment["sentences"] = sentences

    return sentiment


def normalize_text(s):
    """Normalize text for tokenization."""

    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()

    return s

def generate_embeddings_for_call_contents(call_contents):
    """Generate embeddings for call contents. Key assumptions:
    - Call contents is a single string."""

    # Normalize the text for tokenization
    normalized_content = normalize_text(call_contents)
    response = llm_helper.generate_embeddings(normalized_content)

    return response


def save_transcript_to_cosmos_db(transcript_item):
    """Save embeddings to Cosmos DB vector store. Key assumptions:
    - transcript_item is a JSON object containing call_id (int),
        call_transcript (string), and request_vector (list).
    - Cosmos DB endpoint, key, and database name stored in Streamlit secrets."""

    # Create a CosmosClient
    client = CosmosClient(url=cosmos_endpoint, credential=credential)
    # Load the Cosmos database and container
    database = client.get_database_client(cosmos_database_name)
    container = database.get_container_client(cosmos_container_name)

    # Insert the call transcript
    container.create_item(body=transcript_item)


####################### HELPER FUNCTIONS FOR MAIN() #######################
def get_transcription_results():
    if 'file_transcription_results' in st.session_state:
        return st.session_state.file_transcription_results
    elif 'transcription_results' in st.session_state:
        return st.session_state.transcription_results
    else:
        st.error("Please upload an audio file or record a call before proceeding.")
        return None

def perform_audio_transcription(uploaded_file, speech_lang):
    """Generate a transcription of an uploaded audio file."""

    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("Transcribing the call..."):
        all_results = create_transcription_request(uploaded_file, speech_recognition_language=speech_lang)
        return all_results

def perform_compliance_check(include_recording_message, is_relevant_to_topic):
    """Perform a compliance check on a call transcript."""

    tr = get_transcription_results()
    if tr is not None and len(tr) > 0:
        with st.spinner("Checking for compliance..."):
            compliance_results = is_call_in_compliance(tr, include_recording_message, is_relevant_to_topic)
            st.session_state.compliance_results = compliance_results
        st.success("Compliance check complete!")


def perform_extractive_summary_generation(contents_language="en-us"):
    """Generate an extractive summary of a call transcript.
    That is, a summary that extracts key sentences from the call transcript."""

    tr = get_transcription_results()
    if tr is not None and len(tr) > 0:
        # Use st.spinner() to wrap the summarization process.
        with st.spinner("Generating extractive summary..."):
            # Call the generate_extractive_summary function and set
            # its results to a variable named extractive_summary.
            extractive_summary = generate_extractive_summary(tr, contents_language)
            # Save the extractive_summary value to session state.
            st.session_state.extractive_summary = extractive_summary
        st.success("Extractive summarization complete!")


def perform_abstractive_summary_generation(contents_language="en-us"):
    """Generate an abstractive summary of a call transcript.
    That is, a summary that generates new sentences to summarize the call transcript."""

    tr = get_transcription_results()
    if tr is not None and len(tr) > 0:
        # Use st.spinner() to wrap the summarization process.
        with st.spinner("Generating abstractive summary..."):
            # Call the generate_abstractive_summary function and set
            # its results to a variable named abstractive_summary.
            abstractive_summary = generate_abstractive_summary(tr, contents_language)
            # Save the abstractive_summary value to session state.
            st.session_state.abstractive_summary = abstractive_summary
        st.success("Abstractive summarization complete!")


def perform_openai_summary():
    """Generate a query-based summary of a call transcript."""

    tr = get_transcription_results()
    if tr is not None and len(tr) > 0:
        # Use st.spinner() to wrap the summarization process.
        with st.spinner("Generating Azure OpenAI summary..."):
            # Call the generate_query_based_summary function and set
            # its results to a variable named openai_summary.
            openai_summary = generate_query_based_summary(tr)
            # Save the openai_summary value to session state.
            st.session_state.openai_summary = openai_summary
        st.success("Generating Azure OpenAI summary complete!")


def perform_sentiment_analysis_and_opinion_mining():
    """Analyze the sentiment of a call transcript and mine opinions."""

    tr = get_transcription_results()
    if tr is not None and len(tr) > 0:
        # Use st.spinner() to wrap the sentiment analysis process.
        with st.spinner("Analyzing transcript sentiment and mining opinions..."):
            # Call the create_sentiment_analysis_and_opinion_mining_request function and set its results to a variable named sentiment_and_mined_opinions.
            smo = create_sentiment_analysis_and_opinion_mining_request(tr)
            # Save the sentiment_and_mined_opinions value to session state.
            st.session_state.sentiment_and_mined_opinions = smo
        st.success("Sentiment analysis and opinion mining complete!")

def perform_save_embeddings_to_cosmos_db():
    """Save embeddings to Cosmos DB vector store."""

    tr = ' '.join(get_transcription_results())
    if tr is not None and len(tr) > 0:
        # Use st.spinner() to wrap the embeddings saving process.
        with st.spinner("Saving embeddings to Cosmos DB..."):
            # Generate a call ID based on the text.
            # This is for demonstration purposes--a real system should use a unique ID.
            call_id = abs(hash(tr)) % (10 ** 8)
            embeddings = generate_embeddings_for_call_contents(tr)
            transcript_item = {
                "id": f'{call_id}_{uuid.uuid4()}',
                "call_id": call_id,
                "call_transcript": tr,
                "request_vector": embeddings
            }
            save_transcript_to_cosmos_db(transcript_item)
            st.session_state.embedding_status = "Transcript and embeddings saved for this audio."
        st.success("Embeddings saved to Cosmos DB!")



def main():
    """Main function for the call center dashboard."""

    st.write("""
    # Call Center
    This dashboard is intended to replicate some of the functionality of a call center monitoring solution.
    It is not intended to be a production-ready application.
    """)


    st.write("Choose a language for the speech recognition and language processing services.")
    speech_lang = st.selectbox("Select speech language", env_helper.AZURE_SPEECH_RECOGNIZER_LANGUAGES, index=1)

    st.write("## Upload a Call")
    uploaded_file = st.file_uploader("Upload an audio file", type="wav")
    if uploaded_file is not None and ('file_transcription_results' not in st.session_state):
        st.session_state.file_transcription_results = perform_audio_transcription(uploaded_file, speech_lang)
        st.success("Transcription complete!")

    if 'file_transcription_results' in st.session_state:
        st.write(st.session_state.file_transcription_results)


    st.write("## Perform a Live Call")
    st.write("👆 Remember to choose the language for the speech recognition and language processing services.")
    start_recording = stx.button("Record", key="recording_in_progress")
    if start_recording:
        with st.spinner("Transcribing your conversation..."):
            create_live_transcription_request(speech_lang)

    if 'transcription_results' in st.session_state:
        st.write(st.session_state.transcription_results)


    st.write("""
    #### Clear Messages between Calls
    - Select this button to clear out session state and refresh the page.
    - Do this before loading a new audio file or recording a new call.
    """)
    if st.button("Clear messages"):
        if 'file_transcription_results' in st.session_state:
            del st.session_state.file_transcription_results
        if 'transcription_results' in st.session_state:
            del st.session_state.transcription_results
        streamlit_js_eval(js_expressions="parent.window.location.reload()")


    st.write("## Transcription Operations")

    comp, esum, asum, osum, sent, db = st.tabs(["Compliance",
        "Extractive Summary", "Abstractive Summary", "Azure OpenAI Summary",
        "Sentiment and Opinions", "Save to CosmosDB"])

    with comp:
        st.write("### Is Your Call in Compliance?")
        include_recording_message = st.checkbox("Call needs an indicator we are recording it")
        is_relevant_to_topic = st.checkbox("Call is relevant to the hotel and resort industry")
        if st.button("Check for Compliance"):
            perform_compliance_check(include_recording_message, is_relevant_to_topic)
        if 'compliance_results' in st.session_state:
            st.write(st.session_state.compliance_results)
    with esum:
        if speech_lang == "zh-CN":
            contents_language = "zh-hans"
        else:
            contents_language = speech_lang.lower()
        if st.button("Generate extractive summary"):
            perform_extractive_summary_generation(contents_language)
        if 'extractive_summary' in st.session_state:
            st.write(st.session_state.extractive_summary)
    with asum:
        if speech_lang == "zh-CN":
            contents_language = "zh-hans"
        else:
            contents_language = speech_lang.lower()
        if st.button("Generate abstractive summary"):
            perform_abstractive_summary_generation(contents_language)
        if 'abstractive_summary' in st.session_state:
            st.write(st.session_state.abstractive_summary)
    with osum:
        if st.button("Generate query-based summary"):
            perform_openai_summary()
        if 'openai_summary' in st.session_state:
            st.write(st.session_state.openai_summary)
    with sent:
        if st.button("Analyze sentiment and mine opinions"):
            perform_sentiment_analysis_and_opinion_mining()
        if 'sentiment_and_mined_opinions' in st.session_state:
            data = st.session_state.sentiment_and_mined_opinions
            df = pd.DataFrame.from_records(data["sentences"])
            st.table(df)
    with db:
        if st.button("Save embeddings to Cosmos DB"):
            perform_save_embeddings_to_cosmos_db()
        if 'embedding_status' in st.session_state:
            st.write(st.session_state.embedding_status)

if __name__ == "__main__":
    main()
