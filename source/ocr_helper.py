from pathlib import Path
import numpy as np
from PIL import Image
from rapidocr import RapidOCR

_engine = RapidOCR()
_MAX_IMAGE_SIDE = 1800

def extract_ocr_text(image_path: Path) -> dict:
    """Extract text and confidence scores using RapidOCR; does not alter the source image."""
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        if max(image.size) > _MAX_IMAGE_SIDE:
            scale = _MAX_IMAGE_SIDE / max(image.size)
            image = image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
        result = _engine(np.asarray(image))
    texts = getattr(result,"txts",None) or []
    scores = getattr(result,"scores",None) or []
    lines=[]
    for i,value in enumerate(texts):
        if value and str(value).strip():
            score=scores[i] if i<len(scores) else None
            lines.append({"text":str(value).strip(),"score":float(score) if score is not None else None})
    return {"text":"\n".join(x["text"] for x in lines),"lines":lines}
