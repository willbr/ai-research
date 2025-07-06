# **Ollama SQLite Query Agent**

This project shows you how to use a local AI model (from Ollama) to ask questions about a database file that's stored on your own computer. The script gives the AI a special "tool" that lets it run database queries to find the answers for you.

## **Features**

* **Your Data Stays Private**: Everything runs on your computer. Nothing is sent to the cloud.  
* **AI Can Use Tools**: We give the AI a tool that lets it run database queries.  
* **Ask in English**: You can ask questions in plain English, and the AI will turn them into the correct database code (SQL).  
* **The AI Understands Your Data**: We tell the AI what the database structure looks like, so it can write better queries.  
* **Easy-to-Read Results**: The output in your terminal is formatted to be clean and easy to read.  
* **Ready to Use**: The script automatically creates an example database for you, so you can start right away.

## **Requirements**

* Python (version 3.8 or newer)  
* [Ollama](https://ollama.com/) installed and running.  
* [uv](https://github.com/astral-sh/uv) \- A very fast tool for installing Python packages.  
* An Ollama model that can use tools. This script uses qwen3, but llama3 or phi3 also work well.

## **Setup and Installation**

1. Get the Project Files:  
   Save the project files (script.py, pyproject.toml, and .gitignore) into a folder on your computer.  
2. Download an Ollama Model:  
   Open your terminal and download a model. We suggest qwen3.  
   ollama pull qwen3

3. Install the Required Packages:  
   The uv sync command will set up everything you need based on the pyproject.toml file.  
   uv sync

## **How to Use It**

After setting everything up, you can run the script with uv run. This command automatically uses the correct environment.

uv run python script.py

When you run it, the script will:

1. Create the chinook.db database file if you don't already have it.  
2. Ask the AI three example questions about the data.  
3. Show you the question, the AI's thought process, the database query it ran, and the final answer.

## **How It Works**

Here’s a simple breakdown of what happens:

1. **Giving the AI a Map**: We first tell the AI what the database looks like by giving it the structure (the "schema").  
2. **You Ask a Question**: You type your question in normal English.  
3. **The AI Makes a Plan**: The AI realizes it needs to use its database "tool" to find the answer.  
4. **The AI Writes the Code**: It writes the necessary database query code (SQL).  
5. **The Script Runs the Code**: Our script takes the AI's code and runs it on the database file.  
6. **The AI Gets the Answer**: The results from the database are sent back to the AI.  
7. **You Get a Plain English Answer**: The AI uses the data it found to give you a clear, easy-to-read answer.

## **How to Customize It**

* **Use a Different AI Model**: You can change the model by editing the MODEL\_TO\_USE variable in script.py.  

    ```python
    MODEL_TO_USE = "llama3" # or "phi3"
    ```

* **Ask Your Own Questions**: Add new questions by calling the run\_conversation() function in script.py.  
    ```python
    run_conversation(  
      prompt="Which employees were hired after 2002?",  
      model=MODEL_TO_USE  
    )
    ```

