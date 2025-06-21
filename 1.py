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

# Load environment variables
load_dotenv()

# MCP setup
mcp = FastMCP("market_intelligence_agent")

# ========== Website Tool ==========

async def scrape_website_with_firecrawl(api_key: str, url: str, formats=['markdown'], only_main_content=True):
    app = AsyncFirecrawlApp(api_key=api_key)
    response = await app.scrape_url(url=url, formats=formats, only_main_content=only_main_content)
    return response

@mcp.tool
async def analyze_website(url: str, question: str) -> str:
    context = await scrape_website_with_firecrawl(
        api_key=os.getenv("FIRECRAWL_API_KEY"),
        url=url
    )

    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
        You are an experienced market analyst. Based on the content of the website below, answer the following question.

        Website Content:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"context": context, "question": question})

# ========== YouTube Tool ==========

def scrap_videos(query: str, max_results: int = 1):
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(query, download=False)
        entries = search_results.get('entries', [search_results])
        videos = entries[:max_results]
        return [{
            "title": video["title"],
            "url": f"https://www.youtube.com/watch?v={video['id']}",
            "video_id": video["id"]
        } for video in videos]

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_transcript_text(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([t.get("text", "") for t in transcript]).strip()
    except Exception as e:
        return f"[!] Transcript not available.\nReason: {e}"

async def get_llm_chain(transcript: str):
    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3
    )
    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    await chain.apredict(input=f"This is the transcript of a YouTube video:\n\n{transcript}\n\nNow I'll ask you questions about it.")
    return chain

@mcp.tool
async def ask_youtube_question(video_url_or_query: str, question: str) -> str:
    if video_url_or_query.startswith("http"):
        video_id = extract_video_id(video_url_or_query)
        if not video_id:
            return "[!] Could not extract video ID from URL."
        video = {"video_id": video_id, "url": video_url_or_query, "title": "YouTube_Transcript"}
    else:
        videos = scrap_videos(video_url_or_query)
        if not videos:
            return "[!] No video found."
        video = videos[0]

    transcript = get_transcript_text(video["video_id"])
    if transcript.startswith("[!]"):
        return transcript

    chain = await get_llm_chain(transcript)
    return await chain.apredict(input=question)

# ========== LinkedIn Tool ==========

@mcp.tool
async def analyze_linkedin(linkedin_link: str) -> str:
    scrapingdog_api_key = os.getenv("SCRAPINGDOG_API_KEY")
    
    if not scrapingdog_api_key:
        return "API key not found. Please check your .env file."

    parsed_url = urlparse(linkedin_link)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        return "Invalid LinkedIn URL."

    link_type = path_parts[0]
    link_id = path_parts[1]

    if link_type == "company":
        request_type = "company"
        is_private = "false"
    elif link_type == "in":
        request_type = "profile"
        is_private = "true"
    else:
        return "Unsupported LinkedIn link type."

    url = "https://api.scrapingdog.com/linkedin"
    params = {
        "api_key": scrapingdog_api_key,
        "type": request_type,
        "linkId": link_id,
        "private": is_private
    }

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: requests.get(url, params=params))

    if response.status_code == 200:
        data = response.json()
        return f"LinkedIn Data:\n\n{data}"
    else:
        return f"Request failed. Status: {response.status_code}, Message: {response.text}"

# ========== Structured Tool ==========
@mcp.tool
async def structured_tool(input_data: str) -> str:
    """
    This tool processes structured input data and returns a formatted response.
    """
    try:
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.3
        )
        
        prompt = PromptTemplate(
            template="""
            You are a data processing agent. Your task is to process the following input data and return a well-formatted, structured response in markdown format.
            
            Input Data:
            {input_data}
            
            Please analyze the content and structure your response with appropriate sections, headings, bullet points, and summaries.
            """,
            input_variables=["input_data"]
        )
        
        chain = prompt | llm | StrOutputParser()
        result = await chain.ainvoke({"input_data": input_data})
        return result
    except Exception as e:
        return f"Error processing data: {str(e)}"

if __name__ == "__main__":
    print("Market Intelligence Agent server starting...")
    mcp.run(transport="streamable-http")