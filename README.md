# Multi-Agent Research & Report Writer

This repository contains a production-ready agentic application built using raw Python, the Groq API (`llama3-70b-8192`), and the Wikipedia REST API. It serves as the Part 1 submission for the DeepLearning.AI Agentic AI course.

## Overview
The application automates the research and drafting process for academic essays. Given a high-level topic, the system dynamically plans a research strategy, gathers factual information from the web, drafts an essay, and iteratively refines the text through an automated critique loop. 

## Agentic Design Patterns Implemented
This project avoids external frameworks (like LangChain or CrewAI) to demonstrate a fundamental understanding of four core agentic patterns:

1. **Multi-Agent System:** Distributes tasks across four distinct LLM personas (Planner, Researcher, Writer, Critic).
2. **Planning:** The Planner Agent breaks complex topics into actionable, targeted search queries.
3. **Tool Use:** The Researcher Agent programmatically interfaces with the Wikipedia REST API to ground the essay in factual data.
4. **Reflection:** The Critic Agent evaluates drafts against strict criteria, feeding actionable critique back to the Writer Agent in a self-correcting loop until quality thresholds are met.

## Prerequisites
Before running the application, ensure you have Python 3 installed along with the required libraries:

```bash
pip install groq requests

export GROQ_API_KEY="your_api_key_here"
