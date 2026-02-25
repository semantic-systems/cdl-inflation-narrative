import requests
import json
from pathlib import Path
from typing import Dict, List, Optional

# Model mapping
academic_model_map = {
    #'GWDG-meta-llama-3.1-8b-instruct': 'meta-llama-3.1-8b-instruct',
    #"GWDG-mistral-large-instruct": "mistral-large-instruct",
    #"GWDG-qwen3-30b-a3b-thinking-2507": "qwen3-30b-a3b-thinking-2507",
    #'GWDG-llama-3.3-70b-instruct': 'llama-3.3-70b-instruct',
    #"GWDG-openai-gpt-oss-120b": "openai-gpt-oss-120b",
    #"GWDG-gemma-3-27b-it": "gemma-3-27b-it",
    'GWDG-qwen-32b': 'qwen3-32b',
    'GWDG-teuken-7b-instruct-research': 'teuken-7b-instruct-research'
}


def load_prompt(prompt_file: str) -> str:
    """
    Load prompt from a text file.
    
    Args:
        prompt_file: Path to the prompt file
    
    Returns:
        Prompt content as string
    """
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def call_academic_api(
    system_prompt: str,
    user_message: str,
    model: str,
    academic_cloud_api_key: str,
    max_tokens: int = 100,
    temperature: float = 0.1
) -> Dict:
    """
    Call the GWDG Academic Cloud API.
    
    Args:
        system_prompt: System prompt content
        user_message: User message content
        model: Model name (e.g., 'GWDG-deepseek-r1')
        academic_cloud_api_key: API key for authentication
        max_tokens: Maximum tokens in response
        temperature: Temperature for generation
    
    Returns:
        API response as dictionary
    """
    academic_model = academic_model_map.get(model, model.replace('GWDG-', ''))
    #print(f"Calling API: Mapped {model} to {academic_model}")
    
    api_response = requests.post(
        'https://chat-ai.academiccloud.de/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {academic_cloud_api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': academic_model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
    )
    
    if api_response.status_code == 200:
        return api_response.json()
    else:
        raise Exception(f"API request failed with status {api_response.status_code}: {api_response.text}")


def classify_relevance(
    text: str,
    model: str,
    api_key: str,
    prompt_file: str = "./models/prompts/prompt_relevance.txt"
) -> Dict:
    """
    Classify text for inflation cause relevance (0.0-1.0).
    
    Args:
        text: Text to analyze
        model: Model name
        api_key: API key
        prompt_file: Path to relevance prompt file
    
    Returns:
        API response
    """
    system_prompt = load_prompt(prompt_file)
    user_message = f'Analyze this text for inflation cause discussion: "{text}"'
    
    return call_academic_api(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        academic_cloud_api_key=api_key,
        max_tokens=50,
        temperature=0.1
    )


def extract_events(
    text: str,
    model: str,
    api_key: str,
    prompt_file: str = "./models/prompts/prompt_events.txt"
) -> Dict:
    """
    Extract inflation-related events from text.
    
    Args:
        text: Text to analyze
        model: Model name
        api_key: API key
        prompt_file: Path to events prompt file
    
    Returns:
        API response with extracted events
    """
    system_prompt = load_prompt(prompt_file)
    user_message = f'Extract inflation-related events from this text: "{text}"'
    
    return call_academic_api(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        academic_cloud_api_key=api_key,
        max_tokens=500,
        temperature=0.1
    )


def extract_relations(
    text: str,
    events: List[Dict],
    model: str,
    api_key: str,
    prompt_file: str = "./models/prompts/prompt_relations.txt"
) -> Dict:
    """
    Extract causal relationships between events.
    
    Args:
        text: Original text to analyze
        events: List of extracted events from extract_events
        model: Model name
        api_key: API key
        prompt_file: Path to relations prompt file
    
    Returns:
        API response with extracted relationships
    """
    # Load prompt template
    prompt_template = load_prompt(prompt_file)
    
    # Format events for the prompt
    events_str = json.dumps(events, indent=2)
    
    # Replace {events} placeholder with actual events
    system_prompt = prompt_template.replace('{events}', events_str)
    
    user_message = f'Analyze relationships in this text: "{text}"'
    
    return call_academic_api(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        academic_cloud_api_key=api_key,
        max_tokens=800,
        temperature=0.1
    )


def full_pipeline(
    text: str,
    model: str,
    api_key: str,
    relevance_threshold: float = 0.5
) -> Dict:
    """
    Run full analysis pipeline: relevance -> events -> relations.
    
    Args:
        text: Text to analyze
        model: Model name
        api_key: API key
        relevance_threshold: Minimum relevance score to continue analysis
    
    Returns:
        Dictionary with all analysis results
    """
    results = {
        'text': text,
        'relevance': None,
        'events': None,
        'relations': None
    }
    
    # Step 1: Check relevance
    print("Step 1: Checking relevance...")
    relevance_response = classify_relevance(text, model, api_key)
    results['relevance'] = relevance_response
    
    # Parse relevance score (this depends on how the model returns it)
    # You may need to adjust this parsing based on actual response format
    try:
        relevance_score = float(relevance_response['choices'][0]['message']['content'])
        print(f"Relevance score: {relevance_score}")
        
        if relevance_score < relevance_threshold:
            print(f"Text not relevant (score {relevance_score} < {relevance_threshold}). Skipping further analysis.")
            return results
    except (KeyError, ValueError, IndexError) as e:
        print(f"Warning: Could not parse relevance score: {e}")
    
    # Step 2: Extract events
    print("Step 2: Extracting events...")
    events_response = extract_events(text, model, api_key)
    results['events'] = events_response
    
    # Parse events
    try:
        events_content = events_response['choices'][0]['message']['content']
        events_data = json.loads(events_content)
        events_list = events_data.get('events', [])
        print(f"Extracted {len(events_list)} events")
        
        if not events_list:
            print("No events found. Skipping relation extraction.")
            return results
    except (KeyError, ValueError, json.JSONDecodeError, IndexError) as e:
        print(f"Warning: Could not parse events: {e}")
        return results
    
    # Step 3: Extract relations
    print("Step 3: Extracting causal relations...")
    relations_response = extract_relations(text, events_list, model, api_key)
    results['relations'] = relations_response
    
    # Parse and display causal relationships
    try:
        relations_content = relations_response['choices'][0]['message']['content']
        relations_data = json.loads(relations_content)
        causal_rels = relations_data.get('causal_relationships', [])
        print(f"Extracted {len(causal_rels)} causal relationships")
    except (KeyError, ValueError, json.JSONDecodeError, IndexError) as e:
        print(f"Warning: Could not parse causal relationships: {e}")
    
    print("Pipeline complete!")
    return results


# Example usage
if __name__ == "__main__":
    api_key = "your_api_key_here"
    model = "GWDG-deepseek-r1"
    
    # Example 1: Just relevance classification
    text1 = "The Federal Reserve raised interest rates to combat inflation caused by supply chain disruptions."
    
    print("=" * 80)
    print("Example 1: Relevance Classification")
    print("=" * 80)
    result1 = classify_relevance(text1, model, api_key)
    print(json.dumps(result1, indent=2))
    
    # Example 2: Extract events
    text2 = "Rising energy prices and labor shortages have driven up costs. The Fed responded with monetary policy tightening, but inflation expectations remain elevated."
    
    print("\n" + "=" * 80)
    print("Example 2: Event Extraction")
    print("=" * 80)
    result2 = extract_events(text2, model, api_key)
    print(json.dumps(result2, indent=2))
    
    # Example 3: Full pipeline
    text3 = "Supply chain disruptions caused by the pandemic led to higher transportation costs. These rising costs forced companies to increase prices, which in turn raised inflation expectations among consumers and workers."
    
    print("\n" + "=" * 80)
    print("Example 3: Full Pipeline")
    print("=" * 80)
    result3 = full_pipeline(text3, model, api_key, relevance_threshold=0.5)
    print(json.dumps(result3, indent=2))