import os
import re
import asyncio
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastmcp import FastMCP
from firecrawl import AsyncFirecrawlApp
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

# Load .env variables
load_dotenv()

# MCP setup
mcp = FastMCP("market_intelligence_agent")

# ========== TOOLS ==========

@mcp.tool
async def analyze_website(url: str, question: str) -> str:
    context = await AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY")).scrape_url(url)
    prompt = PromptTemplate(
        template="""
        You are an experienced market analyst. Based on the content of the website below, answer the following question.

        Website Content:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )
    chain = prompt | ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3
    ) | StrOutputParser()
    return await chain.ainvoke({"context": context, "question": question})


def extract_video_id(url_or_query: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_query)
    return match.group(1) if match else None


def get_transcript_text(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([t.get("text", "") for t in transcript])
    except Exception as e:
        return f"[!] youtube-transcript-api failed: {e}"


def get_transcript_yt_dlp(video_id: str) -> str:
    try:
        ydl_opts = {
            "quiet": True,
            "writesubtitles": True,
            "skip_download": True,
            "subtitleslangs": ["en"],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            subtitles = info.get("subtitles", {}) or info.get("automatic_captions", {})
            if "en" in subtitles:
                return f"[Transcript via captions]\nURL: {subtitles['en'][0]['url']}"
            else:
                return "[!] No English subtitles found via yt-dlp."
    except Exception as e:
        return f"[!] yt-dlp failed to fetch transcript: {e}"


@mcp.tool
async def ask_youtube_question(video_url_or_query: str, question: str) -> str:
    if video_url_or_query.startswith("http"):
        video_id = extract_video_id(video_url_or_query)
        if not video_id:
            return "[!] Could not extract video ID."
    else:
        ydl = yt_dlp.YoutubeDL({'quiet': True})
        try:
            video_id = ydl.extract_info(f"ytsearch:{video_url_or_query}", download=False)['entries'][0]['id']
        except Exception as e:
            return f"[!] yt-dlp search failed: {str(e)}"

    transcript = get_transcript_text(video_id)
    if transcript.startswith("[!]"):
        fallback = get_transcript_yt_dlp(video_id)
        if fallback.startswith("[!]"):
            return f"[!] Failed to retrieve transcript via both methods:\n{transcript}\n{fallback}"
        transcript = fallback

    chain = ConversationChain(
        llm=ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.3
        ),
        memory=ConversationBufferMemory(return_messages=True)
    )
    await chain.apredict(input=f"Here is the transcript:\n{transcript}")
    return await chain.apredict(input=question)


@mcp.tool
async def analyze_linkedin(linkedin_link: str) -> str:
    key = os.getenv("SCRAPINGDOG_API_KEY")
    parsed = urlparse(linkedin_link)
    parts = parsed.path.strip("/").split("/")
    type_ = "company" if "company" in parts[0] else "profile"
    link_id = parts[1]
    params = {"api_key": key, "type": type_, "linkId": link_id, "private": str(type_ == "profile").lower()}
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, lambda: requests.get("https://api.scrapingdog.com/linkedin", params=params))
    return resp.text if resp.status_code == 200 else f"Error {resp.status_code}: {resp.text}"


@mcp.tool
async def structured_tool(input_data: str) -> str:
    prompt = PromptTemplate(
        template="""
        You are a data processing agent. Your task is to process the following input data and return a well-formatted, structured response in markdown format.

        Input Data:
        {input_data}

        Please analyze the content and structure your response with appropriate sections, headings, bullet points, and summaries.
        """,
        input_variables=["input_data"]
    )
    chain = prompt | ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3
    ) | StrOutputParser()
    return await chain.ainvoke({"input_data": input_data})


# ========== Run MCP Server ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render automatically injects PORT
    print(f"✅ MCP server live on http://0.0.0.0:{port}/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
