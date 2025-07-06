import ollama
import sqlite3
import json
import os
from rich.console import Console
from rich.markdown import Markdown

# --- 0. Rich Console Setup ---
console = Console()

# --- 1. Database Setup ---
DB_FILE = "chinook.db"

def setup_database():
    """Creates and populates a sample SQLite database."""
    if os.path.exists(DB_FILE):
        console.print(f"Database '{DB_FILE}' already exists. Skipping creation.")
        return

    console.print(f"Creating database '{DB_FILE}'...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create employees table
    cursor.execute('''
    CREATE TABLE employees (
        EmployeeId INTEGER PRIMARY KEY,
        LastName TEXT,
        FirstName TEXT,
        Title TEXT,
        ReportsTo INTEGER,
        BirthDate TEXT,
        HireDate TEXT,
        Address TEXT,
        City TEXT,
        State TEXT,
        Country TEXT,
        PostalCode TEXT,
        Phone TEXT,
        Fax TEXT,
        Email TEXT
    )
    ''')

    # Insert sample data
    employees = [
        (1, 'Adams', 'Andrew', 'General Manager', None, '1962-02-18', '2002-08-14', '11120 Jasper Ave NW', 'Edmonton', 'AB', 'Canada', 'T5K 2N1', '+1 (780) 428-9482', '+1 (780) 428-3457', 'andrew@chinookcorp.com'),
        (2, 'Edwards', 'Nancy', 'Sales Manager', 1, '1958-12-08', '2002-05-01', '825 8 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 2T3', '+1 (403) 262-3443', '+1 (403) 262-3322', 'nancy@chinookcorp.com'),
        (3, 'Peacock', 'Jane', 'Sales Support Agent', 2, '1973-08-29', '2002-04-01', '1111 6 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 5M5', '+1 (403) 262-3443', '+1 (403) 262-6712', 'jane@chinookcorp.com'),
        (4, 'Park', 'Margaret', 'Sales Support Agent', 2, '1947-09-19', '2003-05-03', '683 10 Street SW', 'Calgary', 'AB', 'Canada', 'T2P 5G3', '+1 (403) 263-4423', '+1 (403) 263-4289', 'margaret@chinookcorp.com'),
        (5, 'Johnson', 'Steve', 'Sales Support Agent', 2, '1965-03-03', '2003-10-17', '7727B 41 Ave', 'Calgary', 'AB', 'Canada', 'T3B 1Y7', '1 (780) 836-9987', '1 (780) 836-9543', 'steve@chinookcorp.com'),
    ]

    cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', employees)

    conn.commit()
    conn.close()
    console.print("Database setup complete.")

# --- 2. Tool Definition ---
def execute_sql(query: str) -> str:
    """
    Executes a SQL query on the chinook.db database and returns the result.
    The query should be a SELECT statement.
    """
    console.print(f"--- Tool 'execute_sql' called with query: [cyan]{query}[/cyan] ---")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        # Get column names from the cursor description
        colnames = [desc[0] for desc in cursor.description]
        # Format rows as a list of dictionaries for better readability
        result = [dict(zip(colnames, row)) for row in rows]
        conn.close()
        return json.dumps(result)
    except sqlite3.Error as e:
        return f"Error executing SQL query: {e}"

# --- 3. Main Conversation Logic ---
def run_conversation(prompt: str, model: str):
    """
    Runs a conversation with the Ollama model, using the defined tool.
    """
    messages = [{'role': 'user', 'content': prompt}]

    console.print(f"\n> [bold green]User[/bold green]: {prompt}")

    # First, let the model decide if it needs to use a tool
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'execute_sql',
                    'description': 'Execute a SQL query on the local chinook.db SQLite database.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The SQL query to execute.',
                            }
                        },
                        'required': ['query'],
                    },
                },
            }
        ],
    )

    messages.append(response['message'])

    # Check if the model wants to call a tool
    if response['message'].get('tool_calls'):
        tool_call = response['message']['tool_calls'][0]
        function_name = tool_call['function']['name']
        function_args = tool_call['function']['arguments']

        if function_name == 'execute_sql':
            # Call the tool function with the arguments provided by the model
            tool_output = execute_sql(query=function_args.get('query'))

            # Append the tool's output to the conversation history
            messages.append({
                'role': 'tool',
                'content': tool_output,
            })

            # Now, get the final response from the model based on the tool's output
            final_response = ollama.chat(
                model=model,
                messages=messages,
            )
            console.print("\n> [bold magenta]Assistant[/bold magenta]:")
            console.print(Markdown(final_response['message']['content']))
        else:
            console.print(f"Unknown tool requested: {function_name}")
    else:
        # The model responded directly without a tool
        console.print("\n> [bold magenta]Assistant[/bold magenta]:")
        console.print(Markdown(response['message']['content']))


if __name__ == "__main__":
    # Ensure the database exists before running the conversation
    setup_database()

    # --- Configuration ---
    # You can change this to other small models like 'llama3', 'phi3' or 'gemma:2b'
    # Make sure you have pulled the model first with `ollama pull <model_name>`
    MODEL_TO_USE = "qwen3" 

    # --- Run examples ---
    run_conversation(
        prompt="Which employees are located in Calgary? Please list their full names.",
        model=MODEL_TO_USE
    )

    run_conversation(
        prompt="What is the title of Andrew Adams?",
        model=MODEL_TO_USE
    )
    
    run_conversation(
        prompt="Who is the general manager?",
        model=MODEL_TO_USE
    )
