import re

# Step 1: Read the file
file_path = r"/home/pofice/.local/share/fcitx5/rime/sync/19eeb228-5228-4410-ab6e-6f2d17805a9f/rime_ice.userdb.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Step 2: Extract Chinese words using regular expressions
chinese_words = re.findall(r'\t([\u4e00-\u9fff]+)', content)

# Step 3: Write the extracted Chinese words to hotwords.txt
with open('./hotwords.txt', 'w', encoding='utf-8') as hotwords_file:
    for word in chinese_words:
        hotwords_file.write(word + '\n')
