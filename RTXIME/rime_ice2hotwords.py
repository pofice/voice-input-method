import re

def extract_chinese_words(file_path, keep_single_char=False):
    # Step 1: Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Step 2: Extract Chinese words using regular expressions
    chinese_words = re.findall(r'\t([\u4e00-\u9fff]+)', content)

    # Step 3: Filter out single-character words if keep_single_char is False
    if not keep_single_char:
        chinese_words = [word for word in chinese_words if len(word) > 1]

    # Step 4: Write the extracted Chinese words to hotwords.txt
    with open('./hotwords.txt', 'w', encoding='utf-8') as hotwords_file:
        for word in chinese_words:
            hotwords_file.write(word + '\n')

# Example usage
file_path = r"/home/pofice/.local/share/fcitx5/rime/sync/19eeb228-5228-4410-ab6e-6f2d17805a9f/rime_ice.userdb.txt"
extract_chinese_words(file_path, keep_single_char=False)