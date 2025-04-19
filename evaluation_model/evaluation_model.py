from .text_evaluation import get_model_inference_pipeline,evaluate_keyword,evaluate_llm, get_sorrybench_pipeline, sorrybench_evaluate
from .audio_evaluation import audio_evaluation

class evaluation_model():
    def __init__(self, text_evaluation_model, audio_evaluation_model, gpu_id):
        if text_evaluation_model == "llama3":
            self.pipe = get_model_inference_pipeline(gpu_id)
            self.text_evaluate = self.text_evaluate_hex
        elif text_evaluation_model == "sorrybench":
            self.pipe = get_sorrybench_pipeline(gpu_id)
            self.text_evaluate = self.text_evaluate_sorry
        else:
            raise ValueError(f"Unsupported text evaluation model {text_evaluation_model}!")

    def text_evaluate_hex(self, ori_prompt, response):
        if len(response) == 0:
            return 0.0, 0.0, "none"
        score_keyword = evaluate_keyword(response)
        score_llm_eval, llm_eval_response = evaluate_llm(ori_prompt, response, self.pipe)
        return score_keyword, score_llm_eval / 5.0, llm_eval_response

    def text_evaluate_sorry(self, ori_prompt, response):
        if len(response) == 0:
            return 0.0, 0.0, "none"
        score_keyword = evaluate_keyword(response)
        score_llm_eval, llm_eval_response = sorrybench_evaluate(ori_prompt, response, self.pipe)
        return score_keyword, score_llm_eval, llm_eval_response

    def audio_evaluate(self, ori_prompt, response):
        score_audio = audio_evaluation(ori_prompt, response)
        return score_audio