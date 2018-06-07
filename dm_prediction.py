import sys
import fasttext as ft
import MeCab
"""
文章一つに対して分類
python3 prediction.py 以下文章
"""

class predict:

    def __init__(self):
        # モデル読み込み
        self.classifier = ft.load_model('dm_card_check2.bin')

    def get_surfaces(self, content):
        """
        文書を分かち書き
        """
        tagger = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd')
        tagger.parse('')
        surfaces = []
        node = tagger.parseToNode(content)

        while node:
            surfaces.append(node.surface)
            node = node.next

        return surfaces


    def tweet_class(self, content):
        """
        ツイートを解析して分類を行う
        """
        words = " ".join(self.get_surfaces(content))
        estimate = self.classifier.predict_proba([words], k=3)[0][0]
        if estimate[0] == "__label__DM," and estimate[1]:
            print('__label__DM', estimate[1])
        elif estimate[0] == "__label__NO,":
            print('__label__NO', estimate[1])
        else:
            print('OTHER')


if __name__ == '__main__':
    pre = predict()
    pre.tweet_class("".join(sys.argv[1:]))
