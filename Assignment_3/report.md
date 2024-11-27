# 執行環境

本次作業使用 Vscode + Python 進行

# 程式語言

python 3.12.7

# 執行方式

1. 在當前環境安裝以下套件
    
    a. pandas (用作資料操作)

    b. numpy  (用作數學運算)

    c. nltk (用來進行 stemming)

    d. tqdm (視覺化程式進行)

2. 資料位置
    
    將 trainset.txt 以及新聞資料(如，1.txt, 2.txt...)放至 ./data/ 資料夾下。

3. 執行程式

    在 command line 下指令 ```python3 pa3.py```，並在最後產生出 result.csv，該檔案即為測試集的預測結果。

# 作業處理邏輯

- 將 trainset.txt 以及所有檔案讀進來，製作成 dataframe 的形式方便後續操作。
- 根據 trainset.txt 的內容將 dataframe 裡的新聞資料分類成訓練集和測試集。
- 對資料進行文字前處理：
    - 移除文章中的標點符號以及數字
    - 將所有文字都轉成小寫
    - 進行 tokenize 切分文字
    - 移除停用字
    - 進行 stemming
- feature selection：對所有 term 計算 log likelihood ratio，並留下前500個最高 LLR 的 term 做後續的訓練。
- 訓練 Multinomial Bayes Model，得到 prior probability 以及 conditional probability。
- 將上個步驟得到的兩個機率用作 inference。
- 產生最終結果 result.csv 檔案 (F1-Score = 0.96904)