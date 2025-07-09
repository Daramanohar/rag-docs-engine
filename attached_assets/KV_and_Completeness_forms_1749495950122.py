# KV_and_Completeness_forms.py
import re

def extract_kv_and_check(text):
    key_variables = []
    extracted_data = {}

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^-?\s*(.+?)\s*[:|]\s*(.+)", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key not in extracted_data:
                key_variables.append(key)
                extracted_data[key] = value
        else:
            parts = re.split(r"\s*\|\s*", line)
            for part in parts:
                match_nested = re.match(r"(.+?)\s*[:|]\s*(.+)", part)
                if match_nested:
                    key = match_nested.group(1).strip()
                    value = match_nested.group(2).strip()
                    if key not in extracted_data:
                        key_variables.append(key)
                        extracted_data[key] = value

    missing_fields = [key for key, value in extracted_data.items() if not value]

    report = "Extracted Key-Value Pairs:\n"
    for k, v in extracted_data.items():
        report += f"{k}: {v}\n"

    if missing_fields:
        report += f"\n\u26a0\ufe0f Missing Fields: {', '.join(missing_fields)}"
    else:
        report += "\n\u2705 Form appears complete."

    return report
