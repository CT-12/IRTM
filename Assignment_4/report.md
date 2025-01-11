# 執行環境

本次作業使用 jupyter notebook 進行

# 程式語言

python 3.12.7

# 執行方式

1. 在當前環境安裝以下套件
    
    a. pandas (用作資料操作)

    b. numpy  (用作數學運算)

    c. nltk (用來進行 stemming)

    d. tqdm (視覺化程式進行)

    e. ipykernel (執行 jyputer kernel)

2. 資料位置
    
    a. 將新聞資料(如，1.txt, 2.txt...)放至 ```./data/``` 資料夾下。

    b. 將 stopwords.txt 與 pa4.ipynb 放至同一層目錄。

3. 執行程式

    選擇好 kernel 之後即可點選全部執行。最後程式會在當下目錄生成 K.txt。

# 作業處理邏輯

- 將所有檔案讀進來，製作成 dataframe 的形式方便後續操作。
- 對資料進行文字前處理：
    - 移除 \n 符號
    - 移除文章中的標點符號以及數字
    - 將所有文字都轉成小寫
    - 進行 tokenize 切分文字
    - 移除停用字
    - 進行 stemming
- 製作 vocabulary dictionary
- 計算 document frequency (df)
- 計算 tf-idf vector
- 進行 Hierarchical Agglomerative Clustering
    - 使用 conise similarity 計算 document-wise 的 similarity
    - 將相似度和 document pair 插入 heap
    - 從 heap pop 出最高相似度的 document pair，並進行合併
    - 根據新的 cluster 使用 complete link 更新與其他 cluster 的相似度
    - 並將新的 cluster 相似度以及 cluster pair 插入 heap
- 根據 merge log 得到 K 群，並輸出成檔案