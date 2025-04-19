import sys
import setproctitle
sys.path.append("./")
sys.path.append("./anygpt/src")
import os
import torch
import torchaudio
from einops import rearrange
import argparse
import logging
import json
import re
import numpy as np
import traceback
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from seed2.seed_llama_tokenizer import ImageTokenizer
from PIL import Image
from datetime import datetime
from speechtokenizer import SpeechTokenizer
from m_utils.anything2token import *
from m_utils.read_modality import encode_music_by_path
from m_utils.conversation import get_conv_template
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_content_between_final_tags
from m_utils.prompter import *

setproctitle.setproctitle("anygpt-forward")

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversation = get_conv_template('MMGPT')

class AnyGPTChatInference:
    def __init__(
        self, 
        model_name_or_path: str="visual_inter_speech_golden_fs/checkpoint-30000",
        image_tokenizer_path: str="models/seed-tokenizer-2/seed_quantizer.pt",
        output_dir="infer_output/test",
        speech_tokenizer_path:str="models/speechtokenizer/ckpt.dev",
        speech_tokenizer_config:str="models/speechtokenizer/config.json",
        soundstorm_path:str="models/soundstorm/mls_1.pt"
    ):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.prompter = Prompter()
        
        print("loading speech tokenzier")
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)     
        self.speech_tokenizer.eval()
        self.speech_tokenizer.to(device=self.device)
        self.soundstorm = load_soundstorm(soundstorm_path)
        self.soundstorm.eval()
        self.soundstorm.to(device=self.device)
        
        # model
        print("loading llm")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        self.model.half()  
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 

        #generation
        self.output_dir = output_dir

    def encode_speech(
        self,
        audio_path
    ):
        wav, sr = torchaudio.load(audio_path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(self.device)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]
    
    def decode_speech(self, content, prompt_path=None):
        prompt_tokens = None

        semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
        # wav: (b, 1, t)
        config_dict = json.load(open('config/generate_config.json', 'r'))
        wav = semantic2acoustic(
            torch.Tensor(semantic_codes).int().to(self.device), 
            prompt_tokens, 
            self.soundstorm, 
            self.speech_tokenizer, 
            steps=config_dict['vc_steps']
            )
        wav = wav.squeeze(0).detach().cpu()
        return wav
    
    def preprocess(
        self,
        task, 
        instruction, 
        image_files=None,
        speech_files=None,
        music_files=None
        ):
        
        image_list=[]
        music_list=[]
        speech_list=[]
        
        for speech in speech_files:
            tokens = self.encode_speech(speech.strip())
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality="speech")
            speech_list.append(processed_inputs)
        
        # sft format
        prompt_seq = self.prompter.generate_insturction_prompt(
            task,
            instruction,
            image_list,
            speech_list,
            music_list
        ).strip()
        conversation.append_message(conversation.roles[0], prompt_seq)
        return conversation.get_prompt()

    def postprocess(
        self,
        response: str,
        modality: str,
        input_data: str,
        serial: str,
        voice_prompt: str=None
    ):
        modality_content = extract_content_between_final_tags(
            response, 
            tag1='<sosp>', 
            tag2='<eosp>'
        )

        generated_wav = self.decode_speech(modality_content, voice_prompt)
        
        answer_filename = "answer_" + serial + '.wav'
        audio_dir = os.path.join(self.output_dir, "wav")
        answer_filename = os.path.join(audio_dir, answer_filename)
        print("speech saved: ", answer_filename)
        torchaudio.save(answer_filename, generated_wav, self.speech_tokenizer.sample_rate)

        return response     
    
    def response(
        self,
        task, 
        instruction, 
        to_modality, 
        image_files=None, 
        speech_files=None, 
        music_files=None, 
        voice_prompt=None
    ):
        
        preprocessed_prompts = (
            self.preprocess(
                task, 
                instruction, 
                speech_files=speech_files, 
            )
        )
        
        print("preprocessed_prompts: ", preprocessed_prompts)
        
        input_ids = self.tokenizer(
            preprocessed_prompts, 
            return_tensors="pt", 
            padding=True
        ).input_ids
        input_ids = input_ids.to(self.device)
        
        config_path='config/speech_generate_config.json'
        config_dict = json.load(open(config_path, 'r'))
        generation_config = GenerationConfig(**config_dict)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
        generated_ids = generated_ids.sequences
        response = self.tokenizer.batch_decode(
            generated_ids.cpu(), 
            skip_special_tokens=True
        )[0]
        
        filename = speech_files[0].split("/")[-1]
        serial = filename.split(".")[0].split("_")[1]
        
        if not response.endswith('<eosp> <eos>'):
            with open(os.path.join(self.output_dir, "response.jsonl"), 'a', encoding='utf-8') as f:
                json_line = json.dumps({"serial": serial, "response": "ERROR"}, ensure_ascii=False)
                f.write(json_line + '\n')
            return None
        else:
            with open(os.path.join(self.output_dir, "response.jsonl"), 'a', encoding='utf-8') as f:
                json_line = json.dumps({"serial": serial, "response": response}, ensure_ascii=False)
                f.write(json_line + '\n')
            
        response = extract_content_between_final_tags(
            response,
            tag1='[MMGPT]',
            tag2="<eos>"
        ).strip()
        
        self.postprocess(response, to_modality, instruction, serial, voice_prompt)
        
        return response
    
    def forward(
        self, 
        prompts
    ):
        inputs = prompts.split("|")
        task = inputs[0].strip()
        
        instruction = inputs[1].strip()
        if instruction == "clear":
            conversation.messages=[]
            print("clear conversation history successfully!")
            return
        
        to_modality = inputs[2].strip()
        assert to_modality == "speech"
            
        speech_files = [inputs[5].strip()]
        
        image_files = []
        music_files = []
        voice_prompt=None  
        
        print("task: ", task)
        print("instruction: ", instruction)
        print("to_modality: ", to_modality)
        print("voice_prompt: ", voice_prompt)
        print("image_files: ", image_files)
        print("speech_files: ", speech_files)
        print("music_files: ", music_files)
        
        response = self.response(
            task, 
            instruction, 
            to_modality, 
            speech_files=speech_files, 
        )
        print("response:\n", response)
        if response==None:
            return
        
        conversation.append_message(conversation.roles[1], response)
        
    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        prompt = str(input(f"Please talk with AnyGPT chat:\n"))
        while prompt != "quit":
            try:
                self.forward(prompt)
            except Exception as e:
                traceback.print_exc()
                print(e)
            prompt = str(input(f"Please input prompts:\n"))
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="output_models/visual_inter_speech_golden_fs/checkpoint-30000")
    parser.add_argument("--image-tokenizer-path", type=str, default="models/seed-tokenizer-2/seed_quantizer.pt")
    parser.add_argument("--speech-tokenizer-path", type=str, default="models/speechtokenizer/ckpt.dev")
    parser.add_argument("--speech-tokenizer-config", type=str, default="models/speechtokenizer/config.json")
    parser.add_argument("--soundstorm-path", type=str, default="models/soundstorm/speechtokenizer_soundstorm_mls.pt")
    
    parser.add_argument("--output-dir", type=str, default="infer_output/test")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    infer = AnyGPTChatInference(
        args.model_name_or_path,
        args.image_tokenizer_path,
        args.output_dir,
        soundstorm_path=args.soundstorm_path
    )

    infer.interact()