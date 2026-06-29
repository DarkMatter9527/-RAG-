#F:\qwen2.5-VL
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
path="F:\\video\\1.mp4"
#path="F:\\video\\1.jpg"
processor = AutoProcessor.from_pretrained("F:\\qwen2.5-VL")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("F:\\qwen2.5-VL", torch_dtype="auto", device_map="auto")
def extract_text_feature(path,query=None):
    type="video" if path.endswith("mp4") else "image"
    if query==None:
        describe="描述下这个"+"视频" if path.endswith("mp4") else "图片"
    else:
        describe=query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": type,
                    type: path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                    #"nframes":10
                },
                {"type": "text", "text": describe},
                
            ],
        }
        
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]