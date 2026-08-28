import re

file_path = "/media/storage_data/ai2lab/choemj/iclr2027_conference.tex"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(57, 149): # Lines 58 to 149 (0-indexed 57 to 148)
    line = lines[i]
    if line.startswith("% "):
        if "작성 가이드" in line or "해야 할 실험 결과 삽입" in line:
            lines[i] = "\\textbf{" + line[2:].strip() + "}\n\n"
        elif "- " in line:
            lines[i] = "\\begin{itemize}\n\\item " + line[2:].strip().replace("- ", "") + "\n\\end{itemize}\n"
        elif "1." in line or "2." in line or "3." in line or "4." in line or "5." in line:
            lines[i] = line[2:]
        else:
            lines[i] = line[2:]

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
