# Создание уникальной базы знаний для RAG на основе вселенной Star Wars

## 1. Выбор предметной области и подготовка данных

**Вселенная:** Star Wars (каноническая и расширенная вселенная).  
**Объём:** 30+ сущностей, включая персонажей, планеты, технологии и события.  
**Цель:** Заменить все ключевые термины на вымышленные аналоги, чтобы LLM не могла ответить на вопросы по памяти, а полагалась исключительно на загруженные документы.

Для получения исходных текстов использовались публичные страницы фандома (starwars.fandom.com). Каждая сущность была извлечена в виде чистого текста (без HTML-разметки) и сохранена в отдельный файл.
Полностью отличный от исходного текст с данным словарем замен составить не выйдет, но свою задачу он выполняет.

## 2. Словарь замен (terms_map.json)

- Фонетическая и структурная близость – новые имена звучат органично и легко запоминаются (например, Tatooine → Dusthold – намёк на пустынную планету, Obi‑Wan Kenobi → Eldric Vane – сохранение трёхсложной структуры).
- Смысловая ассоциация – замены передают суть оригинала: Jedi → Luminari (от «light»), Sith → Umbra (от «shadow»), Death Star → Oblivion Sphere (сфера уничтожения).

При создании словаря замен руководствовался следующими принципами:При создании словаря замен руководствовался следующими принципами:

```json
{
  "Luke Skywalker": "Kaelen Starfire",
  "Darth Vader": "Vorn Draven",
  "Leia Organa": "Lyra Voss",
  "Han Solo": "Rian Talon",
  "Chewbacca": "Gorrak",
  "Obi-Wan Kenobi": "Eldric Vane",
  "Yoda": "Zorrin",
  "Emperor Palpatine": "Malachar",
  "Boba Fett": "Kael Sharps",
  "Jabba the Hutt": "Gormak the Slug",
  "R2-D2": "Beep-42",
  "C-3PO": "Gold-7",
  "Tatooine": "Dusthold",
  "Alderaan": "Verdantia",
  "Hoth": "Glaciel",
  "Endor": "Woodhaven",
  "Coruscant": "Metropolix",
  "Naboo": "Aquanelle",
  "Mustafar": "Cinderpeak",
  "Kamino": "Tempest",
  "Geonosis": "Aridia",
  "Death Star": "Oblivion Sphere",
  "Millennium Falcon": "Star Drifter",
  "lightsaber": "plasma blade",
  "The Force": "Aetheric Nexus",
  "Jedi": "Luminari",
  "Sith": "Umbra",
  "Wookiee": "Furrok",
  "Hutt": "Slugmorph",
  "droid": "mech-unit",
  "blaster": "ion pistol",
  "hyperdrive": "jump drive",
  "Galactic Empire": "Stellar Dominion",
  "Jedi Master": "Luminari Sage",
  "Padawan": "Initiate",
  "Galactic Republic": "Stellar Concord",
  "Breha Organa": "Elara Voss",
  "Galactic Senate": "Stellar Council",
  "Jango Fett": "Doran Sharps"
}