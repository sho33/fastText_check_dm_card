import fasttext as ft
import MeCab
import sys
import codecs
"""
ファイル一つを読み込んで分類
python3 prediction_file.py test_1.txt
"""

test_list = []
f = codecs.open(sys.argv[1], 'r', 'utf-8')
test_list = f.readlines()
f.close()

m = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
classifier = ft.load_model("dm_card_check2.bin")
for text in test_list:
    items = m.parse(text)
    estimate = classifier.predict_proba([items], k=3)[0][0]
    if estimate[0] == "__label__DM," and estimate[1]:
        print('__label__DM', estimate[1])
    elif estimate[0] == "__label__NO,":
        print('__label__NO', estimate[1])
        print(text)
    else:
        print('OTHER')
    # print(text)
