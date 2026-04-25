import re
import unicodedata

from bs4 import BeautifulSoup


_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


_BULLETS_RE = re.compile(r"[•·●◦▪▫◾◽□■►▶→⇒✓✔✗✘]")


_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_WHITESPACE_RE = re.compile(r"[ \t]+")

MAX_CHARS = 3000


def clean_description(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # нормализуем кодировку, чтобы не было разных артефактов (странных символов)
    text = unicodedata.normalize("NFKC", text)

    # также уберем HTML теги, если такие вдруг случайно спарсились в описание объявлений
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # уберем эмодзи и "буллет" символы (жирные точки, стрелочки и тд..)
    text = _EMOJI_RE.sub(" ", text)
    text = _BULLETS_RE.sub(" ", text)

    #склеим табуляции и пустые строки 
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    # обрезка промпта чтобы уместиться в котекстное окно выбранного эмбеддера
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text
