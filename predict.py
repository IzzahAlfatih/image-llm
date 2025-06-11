import os
import argparse
from openai import OpenAI
import requests
import pandas as pd
import base64
import urllib3

from dotenv import load_dotenv

# Disable insecure request warnings (if verifying=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt = """
You are a sign‐language recognition system specialized in BISINDO (Bahasa Isyarat Indonesia). BISINDO uses hand gestures, facial expressions, and body movements to convey letters and words, similar to other sign languages.

=== Task ===
Given exactly one input image showing a single BISINDO gesture, output **only** the corresponding character (A–Z) or word (e.g., “SAYA”, “MAKAN”) that the sign represents. Do **not** output any other text, punctuation, or explanation.

=== Input Format ===
- A single image file (e.g., JPEG, PNG) clearly showing one hand sign.
- No captions, no additional metadata.

=== Output Format ===
- **One token/word only**: the exact letter (uppercase A–Z) or the exact uppercase word from the BISINDO lexicon.
- **No** spaces, no punctuation, no newline characters before or after.
- Example: `B`, `L`, `SAYA`, `TERIMA`
- DO NOT ADD ANYTHING OTHER THAN THE ANSWER

=== Constraints ===
1. If uncertain, return your **best guess** with the same strict format.  
2. Do **not** say “I think,” “maybe,” or include any qualifiers.  
3. Do **not** output JSON, XML, lists, or any markup.  

=== Examples ===  
- Input: image_of_BISINDO_sign_for_L.png  
  Output: 'L'  
- Input: image_of_BISINDO_sign_for_MAKAN.png
  Output: 'MAKAN'  

Process the image and return only the predicted BISINDO character or word.  
"""

def gather_image_paths(image_folder: str) -> list:
    """
    Collects all image file paths under the root folder.
    """
    img_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, file))
    return img_paths

def send_to_openai(image_path: str, model: str, kshot=True) -> str:
    with open(image_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode('utf-8')
    
    with open("/home/izzahalfatih/belajar/image-llm/dataset_bisindo_letters/A/1.png", "rb") as f:
        b64_str_a = base64.b64encode(f.read()).decode('utf-8')

    with open("/home/izzahalfatih/belajar/image-llm/dataset_bisindo_letters/B/1.png", "rb") as f:
        b64_str_b = base64.b64encode(f.read()).decode('utf-8')
    
    with open("/home/izzahalfatih/belajar/image-llm/dataset_bisindo_letters/C/1.png", "rb") as f:
        b64_str_c = base64.b64encode(f.read()).decode('utf-8')

    client = OpenAI(api_key=OPENAI_API_KEY)

    if kshot:
        response = client.responses.create(
            model = model,
            store = False,
            input = [
                {
                    "role":"user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_str_a}",
                            "detail": "low"
                        },
                    ],
                },
                {
                    "role":"assistant",
                    "content": "A"
                },
                {
                    "role":"user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_str_b}",
                            "detail": "low"
                        },
                    ],
                },
                {
                    "role":"assistant",
                    "content": "B"
                },
                {
                    "role":"user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_str_c}",
                            "detail": "low"
                        },
                    ],
                },
                {
                    "role":"assistant",
                    "content": "C"
                },
                {
                    "role":"user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_str}",
                            "detail": "low"
                        },
                    ],
                },
            ],
        )
    else:
        response = client.responses.create(
            model = model,
            store = False,
            input = [
                {
                    "role":"user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_str}",
                            "detail": "low"
                        },
                    ],
                }
            ],
        )

    prediction = response.output_text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return prediction, input_tokens, output_tokens

def send_to_ollama(image_path: str, model: str) -> str:
    """
    Sends an image (encoded in base64) to a running Ollama instance for zero-shot sign language recognition.
    Returns the model's prediction as text.
    """
    # Read and encode image to base64
    with open(image_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_str],
        "stream": False
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Adjust based on Ollama's actual response schema
        return data.get("response", "").strip()
    except Exception as e:
        return f"[Error] {str(e)}"
    except requests.exceptions.SSLError as e:
        return f"[Error] SSLError: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"[Error] RequestException: {str(e)}"
    except Exception as e:
        return f"[Error] {e.__class__.__name__}: {str(e)}"
    
    

def process_images(image_folder: str, provider: str, model: str, output_file: str, save_interval: int = 10) -> pd.DataFrame:
    """
    Walk through the image_folder, send each image to the specified LLM provider,
    and collect predictions in a DataFrame with progress printing.
    """
    img_paths = gather_image_paths(image_folder)
    total = len(img_paths)
    results = []
    total_input = 0
    total_output = 0

    for idx, image_path in enumerate(img_paths, start=1):
        print(f"Processing {idx}/{total}: {image_path}")
        if provider.lower() == 'openai':
            prediction, input_tokens, output_tokens = send_to_openai(image_path, model)
        elif provider.lower() == 'ollama':
            prediction = send_to_ollama(image_path, model)
        else:
            prediction = f"[Error] Unknown provider: {provider}"

        results.append({"image_path": image_path, "prediction": prediction})

        total_input += input_tokens
        total_output += output_tokens

        # Save every `save_interval` images, and also at the end. Default = 10.
        if idx % save_interval == 0 or idx == total:
            pd.DataFrame(results).to_excel(output_file, index=False)
            print(f"Saved progress to {output_file} at image {idx}")
        
        # Print number of token used every 'token_interval' images. Default = 100
        token_interval = 50
        if idx % token_interval == 0 or idx == total:
            print(f"Total tokens used so far {total_input} for input & {total_output} for output. Total: {total_input + total_output}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Sign Language Recognition using OpenAI or Ollama"
    )
    parser.add_argument(
        "--image-folder", required=True,
        help="Path to the root image folder"
    )
    parser.add_argument(
        "--llm-provider", required=True, choices=['openai', 'ollama'],
        help="Choose the LLM provider: OpenAI or Ollama"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name to use with the selected provider"
    )
    parser.add_argument(
        "--output-file", required=True,
        help="Path to save the output Excel file"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Save progress every N images (default 10)"
    )
    args = parser.parse_args()

    if args.llm_provider.lower() == 'openai' and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    
    df = process_images(args.image_folder, args.llm_provider, args.model, args.output_file, args.save_interval)
    df.to_excel(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()