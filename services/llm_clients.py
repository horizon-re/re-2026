import requests
import time

# ----------------------------- #
#        SUMMARIZATION         #
# ----------------------------- #
def call_glm_z1_summarizer(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5002/glm-z1",
            json={"text": prompt_text}
        )
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"[ERROR] GLM-Z1 Summarizer failed: {e}"

# ----------------------------- #
#        PHASE-WBS (GPT-4.1)   #
# ----------------------------- #
def call_gpt41_model(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5003/gpt-41",
            json={"text": prompt_text}  # OR use "messages": [...] if structured
        )
        print(f"[LLM RAW RESPONSE] {response.text}")
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        print(f"[ERROR] GPT-4.1 API error: {e}")
        return ""



def call_gpt41_model_with_retry(prompt_text: str, retries=3, delay=5) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                "http://localhost:5003/gpt-41",
                json={"text": prompt_text}
            )
            data = response.json()
            if "response" in data:
                return data["response"]
            elif "error" in data:
                raise ValueError(data["error"])
            else:
                raise ValueError("Invalid response format")
        except Exception as e:
            print(f"⚠️ [{attempt}/{retries}] GPT-4.1 failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                return f"[ERROR] Exhausted GPT-4.1 retries: {e}"

# ----------------------------- #
#        FLAT-WBS (GLM-Z1)     #
# ----------------------------- #
def call_glm_z1_flat_wbs(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5004/glm-z1",
            json={"text": prompt_text}
        )
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"[ERROR] GLM-Z1 Flat-WBS failed: {e}"

# ----------------------------- #
#       VALIDATION (OLLAMA)    #
# ----------------------------- #
def call_deepseek_validator(prompt_text: str) -> dict:
    try:
        response = requests.post(
            "http://localhost:5001/validate",
            json={"text": prompt_text},
            timeout=60
        )
        return response.json()
    except Exception as e:
        print(f"[Validator Error] {e}")
        return {
            "verdict": "fail",
            "reasons": ["Validator call failed"],
            "suggested_fixes": [str(e)]
        }


# ----------------------------- #
#     VALIDATION (THUDM Z1)    #
# ----------------------------- #
def call_glm_z1_validator(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5005/glm-z1",
            json={"text": prompt_text}
        )
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"[ERROR] GLM-Z1 Validator failed: {e}"
    

# ----------------------------- #
# FIX AGENT (LLAMA - 70B - Instruct:free)#
# ----------------------------- #

def call_llama_70b_fix_agent(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5006/llama-3.3-70b-instruct",
            json={"text": prompt_text}
        )
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"[ERROR] LLaMA-70B Fix Agent failed: {e}"

    
  
# ----------------------------- #
#       CLAUDE OPUS 4         # 
# ----------------------------- #
    
def call_claude_opus4_model(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5008/claude-opus-4",
            json={"text": prompt_text}  # OR use "messages": [...] if structured
        )
        print(f"[LLM RAW RESPONSE] {response.text}")
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        print(f"[ERROR] Claude Opus 4 API error: {e}")
        return ""


# ----------------------------- #
#      GPT-5 MINI (OR)         #
# ----------------------------- #
def call_gpt5_mini_model(prompt_text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:5007/gpt-5-mini",
            json={"text": prompt_text}
        )
        print(f"[LLM RAW RESPONSE] {response.text}")
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        print(f"[ERROR] GPT-5 Mini API error: {e}")
        return ""
