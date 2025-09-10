import logging
from typing import List
from app.utils.download_punkt import download_punkt
download_punkt()
from nltk.tokenize.punkt import PunktSentenceTokenizer

logger = logging.getLogger(__name__)

def split_text(text: str, max_chunk_chars) -> List[str]:
    text = text.strip()
    if(not text):
        return []
    
    tokenizer = PunktSentenceTokenizer()
    try:
        sentences = tokenizer.tokenize(text)
    except Exception as e:
        logger.warning(f"punkt tokenizer failed: {e}. using fallback split")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if(not sentences):
        return [text]
    
    chunks, current_chunk = [], ""
    for sentence in sentences:
        potential_chunk = (current_chunk + " " + sentence).strip()

        if(len(potential_chunk) <= max_chunk_chars):
            current_chunk = potential_chunk
        else:
            if(current_chunk):
                chunks.append(current_chunk)
            
            if(len(sentence) <= max_chunk_chars):
                current_chunk = sentence
            else:
                words = sentence.split()
                temp_chunk = ""

                for word in words:
                    if(len(temp_chunk + " " + word)<=max_chunk_chars):
                        temp_chunk = (temp_chunk + " " + word).strip()
                    else:
                        if(temp_chunk):
                            chunks.append(temp_chunk)
                        temp_chunk = word

                    current_chunk = temp_chunk

    if current_chunk:
        chunks.append(current_chunk)

    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 10]
    logger.info(f"text split into {len(chunks)} chunks")
    return chunks, sentences