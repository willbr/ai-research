# **AI Painter**

This project uses a local multimodal AI model (via Ollama) to generate images step-by-step based on a user's text prompt. The script acts as an agent, breaking down the user's goal into smaller plans and executing them by generating drawing commands.

## **Features**

* **Iterative Drawing:** Builds an image over multiple steps.  
* **Multi-Agent Personas:** Uses different "personas" (Planner, Drawer, Reviewer) for different stages of the process.  
* **Simple Command API:** Interacts with the AI using a simple line and remove command system.  
* **Configurable:** Allows you to specify the model, number of steps, and API timeout via command-line arguments.

## **Prerequisites**

This project assumes you have a working installation of [Ollama](https://ollama.com/) and [uv](https://github.com/astral-sh/uv).

## **Setup**

1. **Pull a Model:** You will need a multimodal model. llava is a good starting point.  
   ollama pull llava

2. **Install Python Dependencies:** Use uv to install the required libraries.  
   uv pip install ollama Pillow

## **Usage**

Run the script using uv, providing a prompt and optionally specifying the model, number of steps, and timeout.

**Example:**

uv run python paint.py \--prompt "a house with a red roof and a green tree" \--model llava \--steps 5

Generated images will be saved in the output/ directory.

### **Command-Line Arguments**

* \--prompt: (Required) The text description of the image you want to create.  
* \--steps: (Optional) The number of drawing steps. Default is 5\.  
* \--model: (Optional) The Ollama model to use. Default is llava.  
* \--timeout: (Optional) The timeout in seconds for API calls. Default is 120\.
