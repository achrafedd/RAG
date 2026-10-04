from transformers import AutoModelForCausalLM, AutoTokenizer


class Lmachine:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model_name: str = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def dwiha(self, prompt: str) -> None:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=3276802
        )

        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]) :
        ].tolist()

        content = self.tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip("\n")
        return content


if __name__ == "__main__":
    llm = Lmachine()
    result = llm.dwiha("Hello, are you machine?")
    print(result)
