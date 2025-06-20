import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image
import fitz  
from typing import List, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import __version__ as transformers_version
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# MODEL_NAME = "vidore/colqwen2.5-v0.2"
# MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"
MODEL_NAME = "vidore/colqwen2-v0.1"
# MODEL_NAME = "vidore/colpali-v1.2"
# VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

class ColPaliPDFQA:
    def __init__(self, model_name: str = MODEL_NAME, vlm_model_name: str = VLM_MODEL_NAME):

        # Check transformers version
        print(f"Using transformers version: {transformers_version}")
        print(f"model_name: {model_name}")
        print(f"vlm_model_name: {vlm_model_name}")
        if transformers_version < "4.44.0":
            raise ValueError("Please upgrade transformers to version >= 4.44.0 for Qwen2.5-VL support: pip install --upgrade transformers")

        self.device = get_torch_device("auto")
        print(f"Using device: {self.device}")
        
        if (model_name == "vidore/colpali-v1.2"):

            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            
            print('good')
        elif (model_name == "vidore/colqwen2-v0.1"):
            self.model = ColQwen2.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0", 
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(MODEL_NAME)

        else:
            self.model = ColQwen2_5.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0"
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

        self.vlm_model = None
        self.vlm_processor = None
        
        # Try multiple model variants
        model_variants = [
            vlm_model_name,
            "Qwen/Qwen2-VL-2B-Instruct",
        ]
        
        for model_variant in model_variants:
            try:
                print(f"Loading VLM model: {model_variant}")
                print('this is critical ')
                if (model_variant == "Qwen/Qwen2-VL-2B-Instruct"):
                    self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_variant,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    ).eval()
                    
                    self.vlm_processor = AutoProcessor.from_pretrained(
                        model_variant,
                        trust_remote_code=True
                    )
                    
                else:
                    
                    self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_variant,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    ).eval()
                    
                    # Load processor
                    self.vlm_processor = AutoProcessor.from_pretrained(
                        model_variant,
                        trust_remote_code=True
                    )
                
                print(f"Successfully loaded VLM model: {model_variant}")
                self.vlm_model_name = model_variant  
                break
                
            except Exception as e:
                print(f"Error loading {model_variant}: {e}")
                if model_variant == model_variants[-1]: 
                    print("All VLM model variants failed to load")
                    raise RuntimeError(f"Failed to load any VLM model. Last error: {e}")
                else:
                    print(f"Trying next model variant...")
                    continue
        
        self.pdf_embeddings = None
        self.pdf_pages = None
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 150):

        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            # Convert to RGB PIL Image
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        
        doc.close()
        return images
    
    def encode_pages(self, images):

        batch_size = 1
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_images = self.processor.process_images(batch_images).to(self.device)
            
            with torch.no_grad():
                image_embeddings = self.model(**batch_images)
                all_embeddings.append(image_embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def encode_query(self, question: str):

        queries = self.processor.process_queries([question]).to(self.device)
        
        with torch.no_grad():
            query_embeddings = self.model(**queries)
        
        return query_embeddings[0]
    
    def load_pdf(self, pdf_path: str):

        print(f"Loading PDF: {pdf_path}")
        self.pdf_pages = self.pdf_to_images(pdf_path)
        print(f"Converted {len(self.pdf_pages)} pages to images")
        
        print("Encoding pages...")
        self.pdf_embeddings = self.encode_pages(self.pdf_pages)
        print("PDF loaded and encoded successfully!")
    
    def ask_question(self, question: str, top_k: int = 3):

        if self.pdf_embeddings is None:
            raise ValueError("No PDF loaded. Please call load_pdf() first.")
        
        if self.vlm_model is None:
            raise ValueError("VLM model not loaded properly. Cannot answer questions.")
        
        print(f"Processing question: {question}")
        
        # Encode question
        query_embedding = self.encode_query(question)
        
        scores = self.processor.score_multi_vector(
            query_embedding.unsqueeze(0), 
            self.pdf_embeddings
        )[0]
        
        scores = scores.to("cpu")
        
        # Get top-k most similar pages
        top_indices = torch.topk(scores, min(top_k, len(scores))).indices
        
        results = []
        for idx in top_indices:
            page_num = idx.item()
            score = scores[idx].item()
            page_image = self.pdf_pages[page_num]
            
            # Query VLM with the page image and question
            vlm_answer = self.query_vlm(page_image, question)
            
            results.append((page_num + 1, score, page_image, vlm_answer))
        
        return results
    
    def query_vlm(self, image: Image.Image, question: str):
        
        if self.vlm_model is None or self.vlm_processor is None:
            return "Error: VLM model not properly loaded"
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text", 
                            "text": f"Please analyze this PDF page and answer the following question: {question}"
                        }
                    ],
                }
            ]
            
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
        except Exception as e:
            print(f"Method 1 (vision_info) failed: {e}")
            return f"Error processing input: {str(e)}"
                     
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            try:
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.vlm_processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Get only the new tokens
                input_token_len = inputs['input_ids'].shape[1]
                response_ids = generated_ids[:, input_token_len:]
                
                # Decode response
                response = self.vlm_processor.batch_decode(
                    response_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                return response.strip()
                
            except Exception as e:
                print(f"Error during generation: {e}")
                return f"Error generating response: {str(e)}"

def process_vision_info(messages):

    try:
        from qwen_vl_utils import process_vision_info as pvi
        return pvi(messages)
    except ImportError:
        print("qwen_vl_utils not available, using fallback")
        # Extract images from messages
        image_inputs = []
        for message in messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "image" and "image" in content:
                        image_inputs.append(content["image"])
        return image_inputs, None  # No videos

def main():
    try:
        print(f"Attempting to load with embedding model: {MODEL_NAME}")
        qa_system = ColPaliPDFQA(model_name=MODEL_NAME)
        print("Successfully initialized system")
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return
    
    # Load a PDF
    pdf_path = "DeReC-slides.pdf" 
    try:
        qa_system.load_pdf(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return
    
    questions = [
        # "What is the main topic of this document?",
        # "Are there any financial figures mentioned?", 
        # "What conclusions are drawn in this document?",
        "Give me Computational Efficiency readings for L-Defense llama2?"
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        try:
            results = qa_system.ask_question(question, top_k=2)
            
            for page_num, score, page_image, answer in results:
                print(f"Page {page_num} (Score: {score:.3f})")
                print(f"VLM Answer: {answer}")
                print("-" * 30)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()