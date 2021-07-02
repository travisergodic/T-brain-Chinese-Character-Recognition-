# 玉山 T-brain 比賽: 中文字辨識 ![範例 Colab](https://camo.githubusercontent.com/b5854ca9a95be8292a4563a042a62247ed41e657068847d67bb5b4572e46b145/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f2545352541462541362545342542442539432d436f6c61622d79656c6c6f772e7376673f7374796c653d706f706f75742d737175617265)

## 關於本實作

使用[玉山 T-brain比賽資料集](https://tbrain.trendmicro.com.tw/Competitions/Details/14)實現卷積神經網路手寫識識 800 個中文字，並檢測是否為異常輸入，使用 

+ ResNet50
+ DNN
+ ArcFace layer 

技術提升模型整體效果。

 

## 貢獻者

作者: [胡太維](https://github.com/travisergodic)

其他隊員: 蘇大為、陸宏博 

特別感謝: [Dr. YenLinWu](https://github.com/YenLinWu?fbclid=IwAR0yWMI8pvHhRCiIb6oB20auCVp_GhE14NMGfZRxrYm9XzlqPKa0N1-t8Dg)



## 實作範例

[Colab 操作程式碼參考](https://colab.research.google.com/drive/1UdG4uWwLF0-n9Ziff-usPjkOOOO0fvjd?authuser=2)



## 參考資料

1. 比賽資訊: https://tbrain.trendmicro.com.tw/Competitions/Details/14
2. ArcFace 程式碼參考: https://www.kaggle.com/chankhavu/keras-layers-arcface-cosface-adacos
3. ArcFace 論文參考: https://arxiv.org/pdf/1801.07698.pdf