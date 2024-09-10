from nltk.stem import PorterStemmer

def read_file(file_path: str) -> str:
    fd = open(file_path, 'r')
    text = fd.readlines()
    fd.close()

    text = "".join(text)

    return text

def write_file(text: list[str]):
    fd = open("./result.txt", 'w')
    fd.write(" ".join(text))
    fd.close()

def clean_text(text: str) -> str:
    # Remove punctuation and special characters from the text
    remove_symbols = str.maketrans('', '', '.,?"')
    text = text.translate(remove_symbols)

    # Convert all characters to lowercase
    text = text.lower()

    return text

def tokenize(text: str) -> list[str]:
    tokens = text.split()

    return tokens

def stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = list(set([stemmer.stem(token) for token in tokens]))

    return stemmed_tokens

def remove_stopwords(tokens: list[str]) -> list[str]:
    stopwords = read_file("./stopwords.txt").split()
    cleaned_tokens = [token for token in tokens if token not in stopwords]

    return cleaned_tokens

def extract_terms(text: str):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    stemmed_tokens = stem(tokens)
    cleaned_tokens = remove_stopwords(stemmed_tokens)
    write_file(cleaned_tokens)
    

def main():
    text = read_file("./1.txt")
    extract_terms(text)

if __name__ == "__main__":
    main()