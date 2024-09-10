# 執行環境

使用 **vscode** 進行程式的撰寫及執行

# 程式語言

使用 **python 3.12.5**

# 執行方式

1. 請先使用下方指令安裝所需套件：
    
    ```bash
    pip3 install nltk
    ```
    
    (程式使用 `nltk 3.9.1` 的版本進行 porter’s algorithm)
    
2. 安裝好套件後在 terminal 執行下方指令執行程式： 
    
    ```bash
    python3 pa1.py
    ```
    
    執行完後會在當前目錄生成 **result.txt** 檔案
    

# 作業邏輯說明

作業主要的邏輯都在 `extract_terms` 這個函式當中

```python
def extract_terms(text: str):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    stemmed_tokens = stem(tokens)
    cleaned_tokens = remove_stopwords(stemmed_tokens)
    write_file(cleaned_tokens)
```

如上所示：

1. 將讀進來的文章進行清理，主要是將逗號、句號、問號從文章中移除，並把文章內所有文字轉成小寫。
2. 將清理後的文章轉成 tokens，轉成 token 的方式是利用空格將文字進行分割。
3. 對 token 做 stemming，在這邊 `import` 了 `nltk.stem` 的 `PorterStemmer` 對每個 token 做 stemming。
4. 最後移除 stopwords。
5. 將處理完後的 tokens 寫成文字檔輸出。