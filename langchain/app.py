# Dependencies
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Set API Key
os.environ['OPENAI_API_KEY'] = 'api_key'

# App Framework
st.title('📜 ScriptCraft: Your Ultimate YouTube Script Generator 🚀')
prompt = st.text_input('Plug in your prompt here')

# Check if prompt is provided
if prompt:
    # Prompt Templates
    title_template = PromptTemplate(
        input_variables=['topic'],
        template='write me a youtube video title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables=['title', 'wikipedia_research'],
        template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    )

    # Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Llms
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    wiki = WikipediaAPIWrapper()

    # Generate Title
    title = title_chain.run(prompt)
    st.write(title)

    # Wikipedia Research
    wiki_research = wiki.run(prompt)
    st.write(wiki_research)

    # Generate Script
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(script)

    # Display History
    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
