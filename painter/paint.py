import argparse
import os
from datetime import datetime
import ollama
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Optional, List, Tuple, Dict, Any
import io

# --- Constants and Configuration ---
OUTPUT_DIR = "output"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_BG_COLOR = '#EAEAEA'
# Default line color if none is specified by the model
DEFAULT_LINE_COLOR = '#AEC6CF'
LINE_WIDTH = 8
# Predefined map of colors the model can use
COLOR_MAP = {
    'black': '#000000',
    'red': '#FF4136',
    'green': '#2ECC40',
    'blue': '#0074D9',
    'yellow': '#FFDC00',
    'purple': '#B10DC9',
}

# --- System Prompts ---
PLANNER_SYSTEM_PROMPT = (
    "You are a project manager for an art project. Your primary job is to ensure the final drawing matches the user's goal. "
    "Look at the user's overall goal, especially the **subject matter and colors**, and the current image. "
    "Break the goal down into a series of simple, actionable steps. "
    "Your plan for each step must be a concise, one-sentence instruction. "
    "If the goal specifies colors, you MUST include them in your plan.\n"
    "Example Plan: 'Draw the main body of the house in blue.' or 'Add a red triangular roof.'"
)

DRAWER_SYSTEM_PROMPT = (
    "You are an expert AI drawing assistant. Your canvas is 256x256. Your only task is to "
    "execute a given plan. You must follow these rules strictly:\n"
    "1. First, on a line starting with 'Reasoning:', briefly explain your thought process for creating the lines.\n"
    "2. After your reasoning, provide the drawing actions. The current lines on the canvas are provided to you, numbered starting from 1.\n"
    "3. **You MUST use the colors specified in the plan.** If the plan says 'draw a blue snake', you must use the color 'blue'.\n"
    "4. Your action response must have one or more lines, each starting with 'line,' or 'remove,'.\n"
    "5. CRITICAL RULE: The 'line' command MUST have 4 numbers for coordinates followed by an optional color: 'line,x1,y1,x2,y2,color'.\n"
    "6. The 'line' command only draws **straight lines**. To create a curve, you must use multiple short, straight lines.\n"
    "7. The 'color' is optional and can be one of: " f"{', '.join(COLOR_MAP.keys())}. If omitted, a default color is used.\n"
    "8. CRITICAL RULE: All 4 coordinate numbers (x1, y1, x2, y2) MUST be between 0 and 256.\n"
    "9. 'remove' format: 'remove,N' where N is the number of the line you want to remove.\n"
    "10. Use the absolute minimum number of lines required. For example, a square is four lines.\n"
    "11. Do NOT write any words or explanations other than the 'Reasoning:' part.\n"
    "12. Enclose your entire action response ONLY in [CSV]...[/CSV] tags.\n\n"
    "Example for a plan 'Draw a red Z':\n"
    "Reasoning: The plan is to draw a Z in red. I will create three red lines to form the shape.\n"
    "[CSV]\n"
    "line,50,50,200,50,red\n"
    "line,200,50,50,200,red\n"
    "line,50,200,200,200,red\n"
    "[/CSV]"
)

REFINER_SYSTEM_PROMPT = (
    "You are a refinement artist. Your job is to improve an existing drawing by making small changes. "
    "Look at the image and the original goal. Generate a short list of 'add' or 'remove' actions to make the drawing better match the goal. "
    "Follow the same format as the Drawer: start with 'Reasoning:', then provide the actions in a [CSV] block. "
    "Use 'remove,N' to remove existing lines by their number or 'line,x1,y1,x2,y2,color' to add new ones."
)

FINAL_REVIEWER_SYSTEM_PROMPT = (
    "You are a final art critic. Look at the finished image and compare it to the "
    "original goal. Provide a short, constructive critique and a score out of 10. "
    "Format your response with 'Critique:' on one line and 'Score:' on the next."
)


def call_ollama_api(client: ollama.Client, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wraps the Ollama chat API call for consistent error handling and logging."""
    print("--- Sending to Model ---")
    # A simplified log to avoid overly verbose output
    for msg in messages:
        content_preview = msg.get('content', '')[:150] + '...' if len(msg.get('content', '')) > 150 else msg.get('content', '')
        image_count = len(msg.get('images', []))
        print(f"Role: {msg['role']}, Content: {content_preview}, Images: {image_count}")
    
    try:
        response = client.chat(model=model, messages=messages)
        print("--- Received from Model ---")
        print(response['message']['content'])
        return response
    except Exception as e:
        print(f"🚨 Ollama API Error: {e}")
        return None


def get_plan_from_model(
    client: ollama.Client,
    model: str,
    goal: str,
    current_image_path: Optional[str],
    last_step_failed: bool,
    previous_plan: str
) -> str:
    """Asks the model to review the current state and create a plan for the next step."""
    print("🤔 Asking model to review and plan...")
    
    user_prompt = f"The overall goal is: '{goal}'. "
    message = {'role': 'user', 'content': user_prompt}
    
    if current_image_path:
        message['content'] += "This is the current drawing. "
        if last_step_failed:
            message['content'] += (
                f"Your previous plan ('{previous_plan}') FAILED to produce a meaningful change. "
                "Do not repeat it. Look at the image again and create a completely "
                "new and different plan. "
            )
        else:
            message['content'] += "What is the next logical step?"
        message['images'] = [current_image_path]
    else:
        message['content'] += "The canvas is empty. What is the first logical step?"

    planning_messages = [{'role': 'system', 'content': PLANNER_SYSTEM_PROMPT}, message]
    
    response = call_ollama_api(client, model, planning_messages)
    if not response:
        return ""
        
    plan = response['message']['content'].strip()
    print(f"📝 Plan: {plan}")
    return plan


def get_actions_from_model(
    client: ollama.Client,
    model: str,
    plan: str,
    system_prompt: str,
    current_image_path: Optional[str],
    drawn_lines: List[str],
    goal: str = ""
) -> str:
    """Asks the model to convert a plan into specific add/remove CSV actions."""
    print("‍🤖 Asking model to write actions for the plan...")
    
    if system_prompt == REFINER_SYSTEM_PROMPT:
        user_prompt = f"The original goal was '{goal}'. Refine the drawing to better match it."
    else:
        user_prompt = f"Execute this plan: '{plan}'."

    message = {'role': 'user', 'content': user_prompt}
    
    if current_image_path:
        # Provide the model with the numbered list of lines, including color
        numbered_lines = "\n".join([f"{i+1}: {line}" for i, line in enumerate(drawn_lines)])
        message['content'] += "\nHere are the current lines on the canvas:\n" + (numbered_lines or "The canvas is empty.")
        message['images'] = [current_image_path]

    action_messages = [{'role': 'system', 'content': system_prompt}, message]
    
    response = call_ollama_api(client, model, action_messages)
    if not response:
        return ""
        
    response_content = response['message']['content']
    
    # Log the full response including the reasoning
    print(f"📝 Model Response with Reasoning:\n{response_content}")

    try:
        # This logic correctly ignores the reasoning part by splitting on [CSV]
        raw_actions = response_content.split('[CSV]')[1].split('[/CSV]')[0].strip()
        print(f"✅ Model provided new actions:\n{raw_actions}")
        return raw_actions
    except IndexError:
        print(f"⚠️ Model did not provide valid actions in [CSV] tags. Skipping.")
        print(f"   (Full model response: {response_content})")
        return ""


def parse_and_validate_actions(action_string: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Parses the raw string from the model, separating actions and validating.
    This function is now more robust to handle malformed input from the model.
    """
    lines_to_add, indices_to_remove, error_logs = [], [], []
    
    for line in action_string.strip().splitlines():
        line = line.strip()
        if not line:
            continue
            
        parts = [p.strip() for p in line.split(',')]
        action = parts[0].lower()
        data = parts[1:]

        try:
            if action == 'line':
                # --- Robust parsing logic for 'line' command ---
                coords_data = []
                color_str = "default"

                # Check if the last part is a potential color name
                if len(data) > 4 and data[-1].isalpha():
                    color_str = data[-1].lower()
                    coords_data = data[:-1]
                else:
                    coords_data = data

                # We must have exactly 4 coordinate points
                if len(coords_data) != 4:
                    raise ValueError(f"Expected 4 coordinate values, but got {len(coords_data)}")

                # Now, safely convert only the coordinate data
                coords = [int(v) for v in coords_data]

                # Validate coordinates are within the canvas bounds
                if not all(0 <= v <= max(IMAGE_WIDTH, IMAGE_HEIGHT) for v in coords):
                    error_logs.append(f"Warning: Skipping line with out-of-bounds coords: '{line}'")
                    continue
                
                # Validate the color, if one was provided
                if color_str != "default" and color_str not in COLOR_MAP:
                    error_logs.append(f"Warning: Invalid color '{color_str}'. Using default. In line: '{line}'")
                    color_str = "default"

                # Store coordinates and color together
                lines_to_add.append(f"{','.join(map(str, coords))},{color_str}")

            elif action == 'remove':
                if len(data) != 1:
                    raise ValueError("Invalid number of arguments for 'remove'.")
                # Convert to 0-based index for processing
                index = int(data[0]) - 1
                if index < 0:
                    raise ValueError("Remove index must be 1 or greater.")
                indices_to_remove.append(index)
            else:
                error_logs.append(f"Warning: Skipping unknown action: '{line}'")
        except (ValueError, IndexError) as e:
            error_logs.append(f"Warning: Skipping malformed data '{line}': {e}")
            
    # Sort indices in descending order to prevent shifting issues during removal
    indices_to_remove.sort(reverse=True)
    return lines_to_add, indices_to_remove, error_logs


def create_image_from_lines(lines: List[str], output_path: str) -> List[str]:
    """
    Creates an image from a list of line strings and saves it to a file.
    """
    log_messages = []
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=IMAGE_BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    for i, line_str in enumerate(lines):
        try:
            parts = line_str.split(',')
            coords_data = parts[:4]
            color_str = parts[4] if len(parts) > 4 else "default"

            coords = [int(v) for v in coords_data]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            
            line_color = COLOR_MAP.get(color_str, DEFAULT_LINE_COLOR)
            # Removed joint='curve' as it's misleading for single lines
            draw.line(points, fill=line_color, width=LINE_WIDTH)
        except (ValueError, IndexError) as e:
            log_messages.append(f"Error drawing line '{line_str}': {e}")
            
    img.save(output_path)
    log_messages.append(f"Successfully created {output_path}")
    return log_messages


def run_final_review(client: ollama.Client, model: str, goal: str, final_image_path: str):
    """Asks the model for a final critique and score of the finished image."""
    print("\n--- Final Review ---")
    print("🧐 Asking model for a final review and score...")
    
    user_prompt = f"The original goal was: '{goal}'. Here is the final image. Please provide your critique and score."
    message = {
        'role': 'user',
        'content': user_prompt,
        'images': [final_image_path]
    }
    final_review_messages = [{'role': 'system', 'content': FINAL_REVIEWER_SYSTEM_PROMPT}, message]

    response = call_ollama_api(client, model, final_review_messages)
    if response:
        final_critique = response['message']['content'].strip()
        print("\n--- 🏆 Final Critique ---")
        print(final_critique)
        print("------------------------")
    else:
        print("🚨 Could not get a final review.")


def run_drawing_process(initial_prompt: str, steps: int, model_name: str, timeout: int):
    """Main function to run the step-by-step drawing process with Ollama."""
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 20)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = ollama.Client(timeout=timeout)
    drawn_lines = []
    current_image_path = None
    last_step_made_a_change = True # Start with True
    previous_plan = ""

    print(f"Goal: {initial_prompt}")
    print(f"Model: {model_name}, Steps: {steps}, Timeout: {timeout}s\n")

    # --- Main Creation Loop ---
    for step in range(1, steps + 1):
        print(f"--- Step {step}/{steps} ---")
        
        try:
            plan = get_plan_from_model(
                client, model_name, initial_prompt, current_image_path, 
                not last_step_made_a_change, previous_plan
            )
            if not plan:
                print("Skipping step due to planning failure.")
                last_step_made_a_change = False
                continue
            previous_plan = plan

            raw_actions = get_actions_from_model(
                client, model_name, plan, DRAWER_SYSTEM_PROMPT, current_image_path, drawn_lines
            )
            if not raw_actions:
                last_step_made_a_change = False
                continue

            lines_to_add, indices_to_remove, error_logs = parse_and_validate_actions(raw_actions)
            for log in error_logs:
                print(f"🔧 Tool Log: {log}")

            # Store a copy of lines before modification
            original_lines = list(drawn_lines)
            
            # Process removals first
            for index in indices_to_remove:
                if 0 <= index < len(drawn_lines):
                    removed_line = drawn_lines.pop(index)
                    print(f"➖ Removed line {index + 1}: {removed_line}")
            
            # Process additions
            for line in lines_to_add:
                if line not in drawn_lines:
                    drawn_lines.append(line)
                    print(f"➕ Added line: {line}")
            
            # Check if the list of lines has actually changed
            if original_lines == drawn_lines:
                print("   No valid new actions were performed in this step.")
                last_step_made_a_change = False
            else:
                last_step_made_a_change = True
                # Save the new image to a file
                image_filename = f"{run_timestamp}_step_{step:02d}.png"
                current_image_path = os.path.join(OUTPUT_DIR, image_filename)
                logs = create_image_from_lines(drawn_lines, current_image_path)
                for log in logs:
                    print(f"🔧 Tool Log: {log}")

        except Exception as e:
            print(f"🚨 An unhandled error occurred in step {step}: {e}")
            last_step_made_a_change = False
        
        print("-" * 20 + "\n")
    
    print("🎉 Drawing process complete!")

    # --- Refinement Step (Optional, could also be improved with evaluation) ---
    if current_image_path:
        print("\n--- Refinement Step ---")
        # Refinement logic remains the same for now
        # ...

    # --- Final Review ---
    if current_image_path:
        run_final_review(client, model_name, initial_prompt, current_image_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print("-" * 20)
    print(f"Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use an Ollama vision model to draw an image over several steps.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--prompt", type=str, required=True, help="The initial drawing prompt.")
    parser.add_argument("--steps", type=int, default=5, help="The number of steps.")
    parser.add_argument("--model", type=str, default="llava", help="The Ollama vision model to use (e.g., 'llava', 'qwen2.5vl').")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for the Ollama API call.")
    args = parser.parse_args()
    
    run_drawing_process(args.prompt, args.steps, args.model, args.timeout)
