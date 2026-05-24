import os
import json
import requests
from groq import Groq

# Initialize Groq client
# Make sure your GROQ_API_KEY environment variable is set
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------
# HELPER FUNCTION: LLM Call
# ---------------------------------------------------------
def call_llm(system_prompt, user_message, require_json=False):
    """Handles communication with the Groq API."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3 if require_json else 0.7, # Lower temp for strict JSON
        response_format={"type": "json_object"} if require_json else None
    )
    
    content = response.choices[0].message.content
    if require_json:
        return json.loads(content)
    return content

# ---------------------------------------------------------
# PATTERN 2: TOOL USE (Wikipedia Search Tool)
# ---------------------------------------------------------
def search_wikipedia_tool(query):
    """Searches Wikipedia and returns a summary of the top result."""
    print(f"   [Tool Execution] Searching Wikipedia for: '{query}'...")
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['query']['search']:
            # Return the snippet of the top result
            snippet = data['query']['search'][0]['snippet']
            # Clean HTML tags from Wikipedia response
            clean_snippet = snippet.replace('<span class="searchmatch">', '').replace('</span>', '')
            return clean_snippet
        return "No relevant information found."
    except Exception as e:
        return f"Search failed: {str(e)}"

# ---------------------------------------------------------
# PATTERN 4: MULTI-AGENT SYSTEM (Defining Specialized Agents)
# ---------------------------------------------------------

# Agent 1: The Planner (PATTERN 1: PLANNING)
def planner_agent(topic):
    """Breaks down a complex topic into a multi-step research plan."""
    print("\n[Planner Agent] Creating research plan...")
    sys_prompt = "You are a planning agent. Break the user's topic into 3 specific search queries for research. Output JSON with a single key 'queries' containing a list of strings."
    return call_llm(sys_prompt, f"Topic: {topic}", require_json=True)['queries']

# Agent 2: The Researcher (Uses Tools)
def research_agent(queries):
    """Executes the plan by utilizing the search tool."""
    print("\n[Research Agent] Gathering information...")
    research_data = {}
    for query in queries:
        result = search_wikipedia_tool(query)
        research_data[query] = result
    return research_data

# Agent 3: The Writer
def writer_agent(topic, research_data, feedback=None):
    """Drafts or revises the essay based on research and feedback."""
    print("\n[Writer Agent] Drafting content...")
    sys_prompt = "You are an expert academic writer. Write a cohesive, well-structured 3-paragraph essay based on the provided research data."
    
    user_prompt = f"Topic: {topic}\nResearch Data: {json.dumps(research_data)}\n"
    if feedback:
        print("   [Writer Agent] Incorporating reflection feedback...")
        user_prompt += f"\nPrevious Draft Feedback to Address: {feedback}"
        
    return call_llm(sys_prompt, user_prompt)

# Agent 4: The Critic (PATTERN 3: REFLECTION)
def critic_agent(draft, topic):
    """Evaluates the draft and provides actionable feedback for self-correction."""
    print("\n[Critic Agent] Evaluating draft...")
    sys_prompt = """You are a strict editorial critic. Grade the provided essay draft from 1 to 10 based on structure, clarity, and relevance to the topic. 
    Output JSON with two keys: 'score' (integer) and 'feedback' (string detailing exactly what needs to be improved)."""
    
    return call_llm(sys_prompt, f"Topic: {topic}\nDraft:\n{draft}", require_json=True)

# ---------------------------------------------------------
# MAIN WORKFLOW (Orchestrating the Patterns)
# ---------------------------------------------------------
def main():
    topic = input("Enter a topic for the essay: ")
    
    # 1. Planning Phase
    queries = planner_agent(topic)
    
    # 2. Tool Use Phase
    research_data = research_agent(queries)
    
    # 3. Drafting Phase
    draft = writer_agent(topic, research_data)
    
    # 4. Reflection & Self-Correction Loop
    max_revisions = 2
    for attempt in range(max_revisions):
        evaluation = critic_agent(draft, topic)
        score = evaluation['score']
        feedback = evaluation['feedback']
        
        print(f"   -> Draft Score: {score}/10")
        
        if score >= 8:
            print("   -> Score is acceptable. Finalizing essay.")
            break
        else:
            print(f"   -> Score too low. Requesting rewrite. Feedback: {feedback}")
            draft = writer_agent(topic, research_data, feedback)
            
    # Save the final output
    with open("essay.txt", "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\n\n")
        f.write(draft)
        
    print("\n[System] Workflow complete. Final essay saved to 'essay.txt'.")

if __name__ == "__main__":
    main()