import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from Model.chatbot import Chatbot

user_avatar = "asset/user.png"
assistant_avatar = "asset/logo_pln.png"
chatbot = Chatbot()

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-flash")

# Fungsi untuk membuat bubble chat
def chat_message(role, content):
    if role == "user":
        # Render pesan pengguna dengan profil di kanan
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #dcf8c6; padding: 10px 15px; border-radius: 10px; max-width: 60%; text-align: right; color: black;">
                    {content}
                </div>
                <img src="data:image/jpeg;base64,{get_image_as_base64(user_avatar)}" 
                     alt="User" style="margin-left: 10px; width: 40px; height: 40px; border-radius: 50%;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Render pesan asisten dengan profil di kiri
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpeg;base64,{get_image_as_base64(assistant_avatar)}" 
                     alt="Assistant" style="margin-right: 10px; width: 40px; height: 40px; border-radius: 50%;">
                <div style="background-color: #f1f0f0; padding: 10px 15px; border-radius: 10px; max-width: 60%; color: black;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Fungsi untuk mengonversi gambar ke format Base64
def get_image_as_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
# create the app
st.set_page_config(
    page_title="PLN Chatbot",
)
st.title("Welcome to PLN Chatbot")

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_message = "Hai! Saya asisten virtual Anda. Jangan ragu untuk bertanya, dan saya akan berusaha sebaik mungkin untuk memberikan jawaban dan bantuan."
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# render older messages
for message in st.session_state.messages:
    chat_message(message["role"], message["content"])

# Input pengguna
prompt = st.chat_input("Enter your message...")
if prompt:
    # Tambahkan pesan pengguna ke state dan render
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_message("user", prompt)

    # Simulasi respons asisten
    response = chatbot.chatbot_response(gemini_llm=gemini_llm, message=prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    chat_message("assistant", response)

# Tambahkan CSS untuk mendukung mode gelap
st.markdown(
    """
    <style>
    /* Deteksi mode gelap */
    @media (prefers-color-scheme: dark) {
        div[style*="background-color: #dcf8c6"] {
            background-color: #056162 !important; /* Warna bubble untuk mode gelap */
            color: white !important; /* Ubah teks menjadi putih untuk mode gelap */
        }
        div[style*="background-color: #f1f0f0"] {
            background-color: #333333 !important; /* Warna bubble untuk mode gelap */
            color: white !important; /* Ubah teks menjadi putih untuk mode gelap */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)