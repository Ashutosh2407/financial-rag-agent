import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os,re,json,uuid, time
from dotenv import load_dotenv
from langsmith import Client


load_dotenv()

st.set_page_config(page_title="Financial RAG assistant",page_icon="📊", layout="wide")
st.title("📊 Financial Document Intelligence Assistant")

#Session State--------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

def stream_and_parse(url,payload,stream_box):
    full_answer = ""
    full_payload = None
    interrupt_payload = None
    buffer = ""
    with requests.post(url=url,json = payload, stream= True) as response:
            pass




#RAGAS benchmark------------------------------------
with st.expander("📈 Ragas Benchmark"):
    df_eval = pd.read_csv("src/eval/results.csv")
    df_agg = df_eval.groupby("strategy")[["faithfulness", "answer_relevance", "context_precision"]].mean().reset_index()
    strategy_order = ["dense", "sparse", "hybrid", "compression","weaviate"]
    df_agg["strategy"] = pd.Categorical(df_agg["strategy"], categories=strategy_order, ordered=True)
    df_agg = df_agg.sort_values("strategy")
    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.dataframe(df_agg.round(2).set_index("strategy"), use_container_width=True)
    with col_chart:
        fig = px.line(df_agg,x="strategy", y="faithfulness",markers=True, text="faithfulness")
        fig.update_traces(
            line = dict(color="#01696f", width=3),
            marker=dict(size=12, color="#01696f"),
            texttemplate="%{text:.2f}",
            textposition="top center"
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Retrieval Strategy", tickangle=0, showgrid=False),
            yaxis=dict(title="Faithfulness Score", range=[0, 1.1], gridcolor="#e5e5e5"),
            font=dict(family="sans-serif", size=13),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

#Question and Answer

prompt = st.chat_input("Ask a question ...",width=915)
col_chat, col_evidence = st.columns([3,2])

with col_chat:
    db = st.selectbox("Database",["Weaviate", "Pinecone"],width=200)
    st.subheader("Chat")
    container_height = min(max(500, len(st.session_state.chat_history) * 100), 900)
    chat_container = st.container(height=container_height)
    with chat_container:
        #History Replay loop----
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        #Chat Input-----
        if prompt:
            st.session_state.chat_history.append({"role":"user", "content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

    

            #── API Call + Streaming ──────────────────────────────────────────────
            full_payload = None

            with st.chat_message("assistant"):
                full_answer = ""
                stream_box = st.empty()

                buffer = ""
                with requests.post(
                    os.environ.get("BASE_API_URL","http://localhost:8000/query"),
                    json = {
                        "question": prompt,
                        "top_k":5,
                        "db":db.lower(),
                        "thread_id": st.session_state.thread_id
                    }, stream= True
                ) as response:
                    if response.status_code == 422:
                        print("Error")
                        st.error(response.json())
                        st.stop()
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            buffer+= chunk.decode("utf-8")
                            while "\n\n" in buffer:
                                message, buffer = buffer.split("\n\n",1)
                                if message.startswith("data:"):
                                    raw = message[len("data:"):].strip()
                                    try:
                                        payload = json.loads(raw)
                                        if payload.get("type") == "token":
                                            full_answer+=payload.get("content","")
                                            stream_box.markdown(full_answer + "▌")
                                        elif payload.get("type") == "final":
                                            clean = full_answer.split("```")[0].strip()
                                            clean = re.sub(r'`([^`]*)`', r'\1', clean)
                                            clean = clean.replace("$", r"\$")
                                            stream_box.empty()
                                            stream_box.markdown(clean)
                                            full_payload = payload.get("data")
                                    except json.JSONDecodeError:
                                        pass
            if full_answer:
                # Save the assistant reply to chat_history so it persists on the next rerun.
                # We store sources too so per-message evidence is available for future use.
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_payload["answer"],
                    "sources": full_payload.get("sources", [])
                })
            # Show confidence, cost, and token counts below the answer bubble
            st.caption(
                f"⏱ Confidence: {full_payload['confidence']:.2f}  |  "
                f"💰 Cost: ${full_payload['cost_usd']:.5f}  |  "
                f"🔢 Tokens: {full_payload['prompt_tokens']} in / {full_payload['completion_tokens']} out"
            )

            # ── Build Cited Chunks for Evidence Panel ─────────────────────────
            cited_chunks = []
            for citation in full_payload["citations"]:
            # Parse the chunk number out of "[Chunk N]"
                chunk_num = int(citation.replace("[Chunk ","").replace("]",""))
                match = next((s for s in full_payload['sources'] if s["chunk_id"] == chunk_num), None)
                if match:
                    cited_chunks.append(match)

                # ── Evidence Panel ────────────────────────────────────────────────
                with col_evidence:
                    st.subheader("📚 Sources")
                    # Render each cited source with ticker, year, and a text preview
                    for s in cited_chunks:
                        st.markdown(f"**{s['ticker']}** · {s['year'][:4]}")
                        st.caption(s["preview"][:200] + "...")
                        st.divider()

                                    








# question = st.text_input(
#     label="🔍 Ask a question:",
#     placeholder="What were JPMorgan's key risk factors in 2025?",
#     max_chars=200
# )

# if st.button("Ask") and question:
#     col_answer, col_evidence = st.columns([2,1])
#     full_payload = None
#     with col_answer:
#         st.subheader("💬 Answer")

#         with requests.post(os.environ.get("BASE_API_URL","http://localhost:8000/query"),
#                 json={"question": str(question),"top_k":5,"db":db.lower()},
#                 stream=True
#             ) as response:
#             full_answer = ""
#             stream_box = st.empty()
#             buffer = ""
#             for chunk in response.iter_content(chunk_size=None):
#                 if chunk:
#                     buffer+= chunk.decode("utf-8")
#                     while "\n\n" in buffer:
#                         message, buffer = buffer.split("\n\n",1)
#                         if message.startswith("data:"):
#                             raw = message[len("data:"):].strip()
#                             try:
#                                 payload = json.loads(raw)
#                                 if payload.get("type") == "token":
#                                     full_answer += payload.get("content","")
#                                     stream_box.markdown(full_answer + "▌")
#                                 elif payload.get("type") == "final":
#                                     #print("RAW:", repr(full_answer))                          # see exact characters
#                                     clean = full_answer.split("```")[0].strip()
#                                     #print("AFTER SPLIT:", repr(clean))                        # see if split worked
#                                     clean = re.sub(r'`([^`]*)`', r'\1', clean)
#                                     #print("AFTER REGEX:", repr(clean))                        # see if regex worked
#                                     stream_box.empty()
#                                     clean = clean.replace("$","\$")
#                                     stream_box.markdown(clean)
#                                     full_payload = payload.get("data")
#                             except json.JSONDecodeError as e:
#                                 pass
            

#     st.caption(
#             f"⏱ Confidence: {full_payload['confidence']:.2f}  |  "
#             f"💰 Cost: ${full_payload['cost_usd']:.5f}  |  "
#             f"🔢 Tokens: {full_payload['prompt_tokens']} in / {full_payload['completion_tokens']} out"
#         )

#     cited_chunks = []
#     for citation in full_payload["citations"]:
#         chunk_num = int(citation.replace('[Chunk ',"").replace(']',""))
        
#         match = next((s for s in full_payload['sources'] if s["chunk_id"]==chunk_num),None)
#         if match:
#             cited_chunks.append(match)

#     with col_evidence:
#         st.subheader("📚 Sources")
#         for s in cited_chunks:
#             st.markdown(f"**{s['ticker']}** · {s['year'][:4]}")
#             st.caption(s["preview"][:200] + "...")
#             st.divider()