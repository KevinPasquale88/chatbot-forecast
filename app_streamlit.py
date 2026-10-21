import argparse
import sys

import streamlit as st

from chatbot_core import DATASETS, DATASET_LABELS, handle_message, start_session


def _parse_preset_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.dataset


PRESET_DATASET = _parse_preset_dataset()

st.set_page_config(page_title="Clinical Risk Chatbot", page_icon="🩺")
st.title("🩺 Clinical Risk Chatbot")

if "dataset" not in st.session_state:
    st.session_state.dataset = PRESET_DATASET
    st.session_state.messages = []
    if PRESET_DATASET:
        session, greeting = start_session(PRESET_DATASET)
        st.session_state.session = session
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

if st.session_state.dataset is None:
    st.markdown("Scegli il dataset su cui basare la valutazione:")
    choice = st.radio(
        "Dataset", DATASETS, format_func=lambda d: DATASET_LABELS[d], label_visibility="collapsed"
    )
    if st.button("Inizia"):
        session, greeting = start_session(choice)
        st.session_state.dataset = choice
        st.session_state.session = session
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.rerun()
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("Scrivi qui la tua risposta...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    reply, updated_session = handle_message(st.session_state.session, user_text)
    st.session_state.session = updated_session

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

if PRESET_DATASET is None and st.button("Cambia dataset"):
    st.session_state.dataset = None
    st.session_state.messages = []
    st.rerun()
