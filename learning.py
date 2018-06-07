import sys
import fasttext as ft

"""
ラベル2つ以上を持つデータを元にモデルを作成
python3 learning.py model.txt model
"""

argvs = sys.argv
input_file = argvs[1]
output_file = argvs[2]

classifier = ft.supervised(input_file, output_file)
