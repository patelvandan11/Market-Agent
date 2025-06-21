import streamlit as st
from fastmcp import Client
import asyncio

st.title("📊 Market Intelligence Agent with Structured Output")
st.markdown("Select a source tool to analyze, then process its output with the Structured Tool.")

client = Client("http://127.0.0.1:8000/mcp")

async def call_tool(tool_name: str, params: dict):
    async with client:
        result = await client.call_tool(tool_name, params)
        if isinstance(result, list) and hasattr(result[0], "text"):
            return result[0].text
        return result

def call_tool_sync(tool_name: str, params: dict):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(call_tool(tool_name, params), loop)
        return future.result()
    else:
        return asyncio.run(call_tool(tool_name, params))

tool = st.selectbox("Select Source Tool", [
    "Analyze Website",
    "Ask YouTube Question",
    "Analyze LinkedIn"
])

if tool == "Analyze Website":
    url = st.text_input("Website URL")
    question = st.text_area("Question about the website content")
    if st.button("Run Analysis") and url and question:
        with st.spinner("Calling Analyze Website tool..."):
            try:
                raw_output = call_tool_sync("analyze_website", {"url": url, "question": question})
                # st.markdown("### Raw Output:")
                # st.write(raw_output)

                with st.spinner("Processing output with Structured Tool..."):
                    processed = call_tool_sync("structured_tool", {"input_data": raw_output})
                    st.markdown("### Processed Output:")
                    st.write(processed)

            except Exception as e:
                st.error(f"Error: {e}")

elif tool == "Ask YouTube Question":
    video_url_or_query = st.text_input("YouTube Video URL or Search Query")
    question = st.text_area("Question about the video transcript")
    if st.button("Run Analysis") and video_url_or_query and question:
        with st.spinner("Calling Ask YouTube Question tool..."):
            try:
                raw_output = call_tool_sync("ask_youtube_question", {"video_url_or_query": video_url_or_query, "question": question})
                # st.markdown("### Raw Output:")
                # st.write(raw_output)

                with st.spinner("Processing output with Structured Tool..."):
                    processed = call_tool_sync("structured_tool", {"input_data": raw_output})
                    st.markdown("### Processed Output:")
                    st.write(processed)
            except Exception as e:
                st.error(f"Error: {e}")

elif tool == "Analyze LinkedIn":
    linkedin_link = st.text_input("LinkedIn Profile or Company URL")
    if st.button("Run Analysis") and linkedin_link:
        with st.spinner("Calling Analyze LinkedIn tool..."):
            try:
                raw_output = call_tool_sync("analyze_linkedin", {"linkedin_link": linkedin_link})
                # st.markdown("### Raw Output:")
                # st.write(raw_output)

                with st.spinner("Processing output with Structured Tool..."):
                    processed = call_tool_sync("structured_tool", {"input_data": raw_output})
                    st.markdown("### Processed Output:")
                    st.write(processed)
            except Exception as e:
                st.error(f"Error: {e}")
