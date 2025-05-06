import os
import argparse
# import openai
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

def send_to_openai(image_path: str, model: str) -> str:
    return print("belum jadi, malas ngoding")

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
        "prompt": "Identify the meaning of this sign language image. This is BISINDO sign language. Answer with the answer ONLY, just 'A', 'B', 'E', 'eat', 'sit', etc . DO NOT ADD ANY INFORMATION",
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
    
    

def process_images(image_folder: str, provider: str, model: str) -> pd.DataFrame:
    """
    Walk through the image_folder, send each image to the specified LLM provider,
    and collect predictions in a DataFrame with progress printing.
    """
    img_paths = gather_image_paths(image_folder)
    total = len(img_paths)
    results = []

    for idx, image_path in enumerate(img_paths, start=1):
        print(f"Processing {idx}/{total}: {image_path}")
        if provider.lower() == 'openai':
            prediction = send_to_openai(image_path, model)
            break
        elif provider.lower() == 'ollama':
            prediction = send_to_ollama(image_path, model)
        else:
            prediction = f"[Error] Unknown provider: {provider}"

        results.append({"image_path": image_path, "prediction": prediction})

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
    args = parser.parse_args()

    if args.llm_provider.lower() == 'openai' and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    
    df = process_images(args.image_folder, args.llm_provider, args.model)
    df.to_excel(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()