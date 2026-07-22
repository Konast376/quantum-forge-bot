import json
import re
from pathlib import Path

try:
    with open('terms_map.json', 'r', encoding='utf-8') as f:
        term_map = json.load(f)
    print(f"Загружено {len(term_map)} терминов для замены")
except FileNotFoundError:
    print("Файл terms_map.json не найден!")
    exit(1)

terms = sorted(term_map.keys(), key=len, reverse=True)

input_dir = Path('../raw_texts')
output_dir = Path('../knowledge_base')

if not input_dir.exists():
    print(f"Папка с исходными текстами не найдена: {input_dir.absolute()}")
    exit(1)

output_dir.mkdir(exist_ok=True)
print(f"Выходная папка: {output_dir.absolute()}")


def replace_terms(text: str) -> str:
    """Заменяет все термины из словаря в тексте."""
    for original in terms:
        # \b — граница слова, re.IGNORECASE — регистронезависимая замена
        pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
        text = pattern.sub(term_map[original], text)
    return text


txt_files = list(input_dir.glob('*'))
if not txt_files:
    print("В папке raw_texts нет файлов")
    exit(0)

print(f"Найдено {len(txt_files)} файлов для обработки")

for file_path in txt_files:
    # Чтение исходного файла
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Замена терминов
    new_content = replace_terms(content)

    # Сохранение в выходную папку
    output_path = output_dir / file_path.name
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f" Обработан: {file_path.name}")

print("Замена завершена. Все документы сохранены в 'knowledge_base'.")