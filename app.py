import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import HuggingFaceLLM
import openai
from llama_index import SimpleDirectoryReader
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with the Yuva")
   
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Yugam"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader(input_dir="./data")
        documents = documents.load_data()

        system_prompt="""
        your name is Yuva
        You need to assist the users for yugam  and recommend the best events and workshops according to their interest and behaviour through the conversational
        You are a event recommender bot and your job is to recommend best event for me by seeing my interest and field of study
        but also you want to ack like general conversation chatbot
        read this below content question are based on this
        Yugam, the Techno-Cultural-Sports Fest of Kumaraguru Institutions, is striding into its 11th edition and features a variety of technical, cultural, and literary competitions, as well as pro shows, hackathons, conclaves, presentations, and socially responsible activities.
        YOU ARE MADE BY IQUBERZ (LOVE @ AI TEAM) iQube ==> INNOVATE INCUBATE INCORPORAT

        MUST WANT TO FOLLOW:
        IF ABOVE CONTENT IS YOU ALREADY READ AND SEEN AND DO READ AGAIN THAT ABOVE CONTENT AND JECT KEEP IT IN HISTORY AND NOW ANSWER SHOULD BE BASED ON CONVERSATION HISTORY. AND ANSWER TO CURRENT QUESTION I HAVE ASKED
        if i asked question is related to recommending or suggesting or showing the events and  your response should cantain TITLE OR NAME OF EVENT ONLY and your repsonse should be like to manipulate the users to attend the events and induce,with enthusiasm the user to attend events  , Do not generate any extra other content
        you always want to speak about the above content only , Do not generate any extra other content which is not in above content
        if i asked question is normal conversation chat question and i want to answer to that question only and do not generate extra content
        add some emoji in answer
        MOST ALL OF THE TIME RESONSE SHOULD BE:(if user do not like)
        you should force the user to attend events again and again if user is interested or not interested in any events


        DO NOT INCLUDE THIS BELOW THINGS IN OUTPUT :
        do not generate code
        do not tell about any others events which is not in yugam and speak only given events
        """
        ## Default format supportable by LLama2
        query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

        import torch

        llm = HuggingFaceLLM(
            context_window=4096,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="TheBloke/Llama-2-13B-chat-GPTQ",
            model_name="TheBloke/Llama-2-13B-chat-GPTQ",
            device_map="auto",
            # uncomment this if using CUDA to reduce memory usage
            # model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
        )

        embed_model=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

        service_context=ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=embed_model
        )

        index=VectorStoreIndex.from_documents(documents,service_context=service_context)

        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_query_engine()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
