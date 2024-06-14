import streamlit as st
import settings

st.title("📝 석리송의 ChatGPT")

config = settings.load_config()
if "api_key" in config:
    st.session_state.api_key = config["api_key"]

main_text = st.empty()


api_key = st.text_input("🔑 새로운 OPENAI API Key", type="password")

save_btn = st.button("설정 저장", key="save_btn")

if save_btn:
    settings.save_config({"api_key": api_key})
    st.session_state.api_key = api_key
    st.write("설정이 저장되었습니다.")
