import pandas as pd

def build_prompt(csv_path: str, prompt_template_path: str) -> str:
    # Load merchant data
    df = pd.read_csv(csv_path)

    # Use the first merchant as an example
    merchant_info = df.iloc[0].to_dict()

    # Load prompt template
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Fill template with merchant data
    prompt = template.format(**merchant_info)

    return prompt
