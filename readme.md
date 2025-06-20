## 2 Stage Retrieval Pipeline using ColPali and Vision Language Model (VLM)

Traditional document processing pipelines rely heavily on OCR → text extraction → language models. But what happens when we let multimodal AI directly "see" and interpret documents like ColPali?

We rely extensively on OCR and Layout LLMs to extract relevant information from PDFs and documents, but how effective are newer architecture models like ColPali?

To test this, I gave it a 23-page PDF and asked: "Give me Computational Efficiency readings for L-Defense llama2?"

The answer was embedded in visual charts on specific pages. For it to answer effectively, it had to:

✅ Process my question (query)
✅ Understand the entire 23-page PDF visually
✅ Locate the relevant charts/tables and rxtract precise numerical data
✅ Tell me exactly which pages contained the answer


The Results? Impressive. 📊

The system found the data on pages 18 & 19, extracting detailed runtime metrics:

Evidence Extraction: 61m 39s
LLM-generated Explanations: 381m 31s
Total Runtime: 454m 12s

The Pipeline:

[2 Stage Pipeline using ColPali](images/pipeline-diagram.png)

I used ColQwen2 (vidore/colqwen2-v0.1) combined with Qwen2.5-VL-3B-Instruct in a two-stage approach:

ColQwen2 converts PDF pages to embeddings and finds most relevant pages via similarity search
Qwen2.5-VL analyzes the top-ranked page images and answers the question


The Results: Surprisingly accurate! 🎯

It successfully located pages 18-19 and extracted precise performance metrics:

Evidence Extraction: 61m 39s
LLM-generated Explanations: 381m 31s
Total Runtime: 454m 12s

Exact page references for verification


Why ColPali is interesting:

Unlike traditional OCR → text extraction → LLM pipelines, ColPali treats document pages as images and creates rich multimodal embeddings that capture both textual content AND visual layout. This means it can understand Ccomplex tables and charts and spatial relationships between elements.

ColPali's foundation combines a text encoder (ColBERT's multi-vector retrieval model) and PaliGemma's vision-language model. Recent versions of ColPali utilize Qwen2.5-VL for the vision model which provides better results compared to PaliGemma. 

More about ColPali https://arxiv.org/abs/2407.01449


## Installation 

```
uv venv venv --python 3.10

source venv/bin/activate

uv pip install colpali-engine pymupdf pillow torch transformers matplotlib opencv-python numpy qwen_vl_utils git+https://github.com/illuin-tech/colpali 

```