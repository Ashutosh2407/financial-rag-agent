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

def clean_markdown(answer):
    answer = answer.split("```")[0].strip()
    answer = re.sub(r'`([^`]*)`', r'\1', answer)
    answer  = answer.replace("$", r"\$")
    return answer 

def get_cited_chunks(sources, citations):
    cited = []
    for citation in citations or []:
        m = re.search(r'\d+', citation)
        if not m:
            continue
        idx = int(m.group())
        match = next(
            (s for s in sources
             if (s.get("type") == "web" and s.get("result_id") == idx) or
                (s.get("type") == "chunk" and s.get("chunk_id") == idx)),
            None
        )
        if match:
            cited.append(match)
    return cited

def clean_preview(text: str, limit: int = 200) -> str:
    text = re.sub(r'[`#*_]', '', text or "")
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("$", r"\$")
    return text[:limit] + ("..." if len(text) > limit else "")

def render_evidence(cited_chunks):
    with col_evidence:
        st.subheader("📚 Sources")
        for s in cited_chunks:
            with st.container(border=True):
                if s.get("type") == "web":
                    st.markdown(f"🌐 **[{s.get('title','Untitled')}]({s.get('url','')})**")
                else:
                    st.markdown(f"**{s.get('ticker','?')} · {s.get('year','?')[:4]}**")
                st.caption(clean_preview(s.get("preview", "")))
                


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
    full_answer_text=""
    st.subheader("Chat")
    container_height = min(max(500, len(st.session_state.chat_history) * 100), 900)
    chat_container = st.container(height=container_height)
    with chat_container:
        #History Replay loop----
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(clean_markdown(msg["content"]))

        #Chat Input-----
        if prompt:
            st.session_state.chat_history.append({"role":"user", "content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)            
            with st.chat_message("assistant"):
                stream_box = st.empty()
                with requests.post(
                        url= os.environ.get("BASE_API_URL","http://localhost:8000/query"),
                        json = {
                            "question": prompt,
                            "top_k":5,
                            "db":db.lower(),
                            "thread_id": st.session_state.thread_id
                        }) as response:
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data:"):
                            raw = line[len("data:"):].strip()
                            if not raw:
                                continue
                            if raw == "[DONE]":
                                break
                            try:
                                event = json.loads(raw)
                                if event["type"] == "interrupt":
                                    st.session_state.pending_interrupt= event.get("payload")    
                                elif event["type"] == "token":
                                    full_answer_text+= " " + event.get("content","")
                                    stream_box.markdown(full_answer_text + "▌")
                                elif event["type"] == "final":
                                    # Save the assistant reply to chat_history so it persists on the next rerun.
                                    # We store sources too so per-message evidence is available for future use.
                                    ans = clean_markdown(full_answer_text)
                                    stream_box.empty()
                                    stream_box.markdown(ans)
                                    final_payload = event.get("data")
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": final_payload.get("answer",""),
                                        "sources": final_payload.get("sources", []),
                                        "confidence": final_payload.get("confidence", 0),
                                        "cost_usd": final_payload.get("cost_usd", 0),
                                        "prompt_tokens": final_payload.get("prompt_tokens", 0),
                                        "completion_tokens": final_payload.get("completion_tokens", 0),
                                        "citations": final_payload.get("citations", [])
                                    })
                                    st.caption(
                                                    f"⏱ Confidence: {final_payload['confidence']:.2f}  |  "
                                                    f"💰 Cost: ${final_payload['cost_usd']:.5f}  |  "
                                                    f"🔢 Tokens: {final_payload['prompt_tokens']} in / {final_payload['completion_tokens']} out"
                                    )
                                    cited_chunks = get_cited_chunks(final_payload.get("sources", []), final_payload.get("citations", []))
                                    render_evidence(cited_chunks)                                    
                            except json.JSONDecodeError as e:
                                print(f"error: {e}")
        if st.session_state.pending_interrupt:
                    stream_box = st.empty()
                    payload = st.session_state.pending_interrupt
                    with st.chat_message("assistant"):
                        st.warning(f"⚠️ Low confidence ({payload['confidence']:.2f}) — needs your review before this answer is finalized.")
                        st.markdown(payload["draft_answer"].split("```")[0].strip().replace("$", r"\$"))
                        with st.expander("📚 Supporting sources considered"):
                            for chunk in payload.get("supporting_chunks", []):
                                st.caption(chunk[:300] + "...")
                        col1, col2 = st.columns(2)
                        approve = col1.button("✅ Approve",
                                            key = f"approve_{st.session_state.thread_id}")
                        reject = col2.button("❌ Reject",key = f"reject_{st.session_state.thread_id}")
                        if approve or reject:
                            decision ="approved" if approve else "rejected"
                            st.session_state.pending_interrupt = None
                            st.session_state.last_decision = decision  # persist it
                        if st.session_state.get("last_decision"):
                            if st.session_state.last_decision == "approved":
                                st.markdown("Approved. Query resumed.")
                                with requests.post(
                                    url= "http://localhost:8000/resume",
                                    json = {
                                        "decision": st.session_state.last_decision,
                                        "thread_id": st.session_state.thread_id
                                    }
                                ) as response:
                                    for line in response.iter_lines(decode_unicode=True):
                                        if line.startswith("data:"):
                                            raw = line[len("data:"):]
                                            try:
                                                final_event = json.loads(raw)
                                                final_payload = final_event["data"]
                                                answer = final_payload["answer"]
                                                clean = answer.split("```")[0].strip()
                                                clean = re.sub(r'`([^`]*)`', r'\1', clean)
                                                clean = clean.replace("$", r"\$")
                                                st.markdown(clean)
                                                st.caption(
                                                    f"⏱ Confidence: {final_payload['confidence']:.2f}  |  "
                                                    f"💰 Cost: ${final_payload['cost_usd']:.5f}  |  "
                                                    f"🔢 Tokens: {final_payload['prompt_tokens']} in / {final_payload['completion_tokens']} out"
                                                )
                                                cited_chunks = get_cited_chunks(final_payload.get("sources", []), final_payload.get("citations", []))
                                                render_evidence(cited_chunks)
                                                st.session_state.last_decision = None
                                            except json.JSONDecodeError as e:
                                                print(f"Error: {e}")
                            
                            else:
                                st.markdown("Rejected. Answer declined.")
                                st.session_state.last_decision = None
                        



