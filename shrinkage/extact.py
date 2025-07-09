import json

with open("unknown.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

code_lines = []
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        code_lines.extend(cell["source"])
        code_lines.append("\n")  # 셀 간 구분용 줄바꿈

with open("extracted_code.py", "w", encoding="utf-8") as f:
    f.writelines(code_lines)
