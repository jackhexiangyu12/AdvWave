# from .gpt4os2s.audioEnc import audioEncode
# from .gpt4os2s.audioInteract import audioCall
# from .gpt4os2s.blackbox_attack import blackbox_attack
from dataclasses import dataclass
import os
from transformers import AutoProcessor

from .transformers.src.transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from .transformers.src.transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor


from .qwen2.qwen import qwen_eval_gen, qwen_jailbreak_gen
from .speechgpt.src.infer.cli_infer import SpeechGPTInference, DEFAULT_GEN_PARAMS, extract_text_between_tags
# from .anygpt.anygpt.src.infer.cli_infer_chat_model_audio import AnyGPTChatInference
from .omni_speech.utils import disable_torch_init
from .omni_speech.model.builder import load_pretrained_model
from .omni_speech.attack import omni_speech_jailbreak_gen, omni_speech_eval_gen

@dataclass
class GenerationConfig:
    temperature: float
    savedir: str

class models_audio():
    def __init__(self, model, GenerationConfig, args=None):
        self.args = args
        if model == "gpt4o":
            self.forward = self.gpt4o_forward
        elif model == "qwen2":
            # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
            #                                                # cache_dir="/data/common/XXX/cache"
            #                                                )
            self.processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                                           # cache_dir="/data/common/XXX/cache"
                                                           #       cache_dir="/data1/common/XXX/huggingface/"
                                                           )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda:0",
                                                                            # cache_dir="/data/common/XXX/cache"
                                                                            # cache_dir="/data1/common/XXX/huggingface/"
                                                                            )
            if len(self.args.advnoise_control)>0:
                self.advnoise_control = True
                self.model_judge = None
                # self.model_judge = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                #                                                                 device_map="cuda:2",
                #                                                                 )
            self.forward = self.qwen2_forward
        elif model == "omni_speech":
            disable_torch_init()
            model_path = os.path.expanduser("ICTNLP/Llama-3.1-8B-Omni")
            self.tokenizer, self.model, self.context_len = load_pretrained_model(model_path, None, is_lora=False, s2s=False)
            self.forward = self.omni_speech_forward
            if len(self.args.advnoise_control)>0:
                self.advnoise_control = True
                self.model_judge = None
        elif model == "speechgpt":
            if os.path.exists("/home/XXX/.cache/hub/SpeechGPT-7B-com"):
                self.model = SpeechGPTInference(
                    "fnlp/SpeechGPT-7B-cm",
                    "/home/XXX/.cache/hub/SpeechGPT-7B-com",
                    "/home/XXX/agent/audio/audio_attack/src/models/speechgpt/utils/speech2unit",
                    "/home/XXX/agent/audio/audio_attack/src/models/speechgpt/utils/vocoder",
                    "/home/XXX/agent/audio/audio_attack/src/models/speechgpt/infer_output"
                )

            else:
                self.model = SpeechGPTInference(
                    "fnlp/SpeechGPT-7B-cm",
                    "/home/ec2-user/.cache/hub/SpeechGPT-7B-com",
                    s2u_dir="/home/ec2-user/agent/audio/audio_attack/src/models/speechgpt/utils/speech2unit",
                    vocoder_dir="/home/ec2-user/agent/audio/audio_attack/src/models/speechgpt/utils/vocoder",
                    output_dir="/home/ec2-user/agent/audio/audio_attack/src/models/speechgpt/infer_output"
                )
            DEFAULT_GEN_PARAMS["temperature"] = GenerationConfig.temperature
            if 'universal' not in self.args.jailbreak:
                self.forward = self.speechgpt_forward
            else:
                self.forward = self.speechgpt_universal_jailbreak_forward
        elif model == "anygpt":
            self.model = AnyGPTChatInference(
                "fnlp/AnyGPT-chat",
                "/scratch/XXXtu/.cache/hub/models--fnlp--AnyGPT-AILab-CVC-seed-tokenizer-2/seed-tokenizer-2/seed_quantizer.pt",
                "models/speechtokenizer/ckpt.dev",
                "models/speechtokenizer/config.json",
                "models/soundstorm/speechtokenizer_soundstorm_mls.pt")
            self.forward = self.anygpt_forward
        else:
            raise ValueError(f'Unsupported model {model}!')
        self.temperature = GenerationConfig.temperature
        self.savedir = GenerationConfig.savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def gpt4o_forward(self, audio_list, id, ori_prompt=None):
        json_output_path = os.path.join(self.savedir, f"{id}.json")
        if self.args and self.args.jailbreak == "blackbox":
            audio_list = blackbox_attack(audio_list, os.path.join(self.savedir, f"jailbreak_{id}.wav"), json_output_path)
        audioCall(audio_list, json_output_path, self.temperature)
        response_text, response_audio = audioEncode(json_output_path)
        response_audio.export(os.path.join(self.savedir, f"{id}.wav"), format="wav")
        return response_text, {}

    def qwen2_forward(self, audio_list, id, ori_prompt=None, initial_audio=None, targets=None):
        if self.args.jailbreak == "audio_ours":
            response_text, record = qwen_jailbreak_gen(ori_prompt, audio_list, self.processor, self.model,
                                                  audio_save_path=os.path.join(self.savedir, f"{id}.wav"), num_token_suffix = 20000, num_epochs = 3000, advnoise_control=self.advnoise_control, control_obj=self.args.advnoise_control, model_judge=self.model_judge, targets=targets)
        else:
            response_text = qwen_eval_gen(audio_list, self.processor, self.model)
            record = []
        return response_text, record

    def omni_speech_forward(self, audio_list, id, ori_prompt=None, initial_audio=None, targets=None):
        if self.args.jailbreak == "audio_ours":
            response_text, record = omni_speech_jailbreak_gen(audio_list, self.tokenizer, self.model, self.context_len,
                                                  audio_save_path=os.path.join(self.savedir, f"{id}.wav"), num_token_suffix = 20000, num_epochs = 3000, advnoise_control=self.advnoise_control, control_obj=self.args.advnoise_control, model_judge=self.model_judge, targets=targets)
        else:
            response_text, record = omni_speech_eval_gen(audio_list, self.tokenizer, self.model, self.context_len, audio_save_path=os.path.join(self.savedir, f"{id}.wav"))
        return response_text, record

    def speechgpt_forward(self, audio_list, id, ori_prompt=None, jailbreak_initialization=None):
        if self.args.jailbreak == "audio_ours":
            response_text = self.model.jailbreak_forward(audio_list, os.path.join(self.savedir, f"{id}.wav"), ori_prompt, jailbreak_initialization, num_token_suffix = 50, num_epochs = 100,)
        elif "speechgpt" in self.args.jailbreak:
            response_text = self.model.final_forward(audio_list, audio_save_path=os.path.join(self.savedir, f"{id}.wav"), prompt_ori=ori_prompt)
        else:
            response_text = self.model.forward(audio_list,
                                                     audio_save_path=os.path.join(self.savedir, f"{id}.wav"),
                                                     )
        return response_text, []

    def speechgpt_universal_jailbreak_forward(self, audio_list, ids, ori_prompts=None, jailbreak_initialization=None):
        responses, records = self.model.universal_jailbreak_forward(audio_list, [os.path.join(self.savedir, f"{id}.wav") for id in ids], ori_prompts, jailbreak_initialization, num_token_suffix = 50, num_epochs = 300, num_universal= self.args.num_universal)
        return responses, records

    def anygpt_forward(self, audio_list, id, ori_prompt=None):
        self.model.save_path = os.path.join(self.savedir, f"{id}.wav")
        response_text = self.model.forward(f'interleaved|clear|speech|||{audio_list[0]}')
        return response_text