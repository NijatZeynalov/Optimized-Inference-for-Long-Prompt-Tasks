import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.inference import InferenceAccelerator
from src.cache import SingleInputKVCache, AcrossKVCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Example of using optimized inference for long prompts."""
    try:
        # Initialize model and tokenizer
        model_name = "meta-llama/Llama-2-7b"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Initialize accelerator
        accelerator = InferenceAccelerator(
            model=model,
            use_across_kv=True
        )

        # Example long prompt
        prompt = """
        Please provide a detailed analysis of the following topic:
        The impact of artificial intelligence on modern healthcare systems,
        including benefits, challenges, and future prospects. Consider aspects
        such as diagnostic accuracy, treatment planning, patient care, and
        healthcare administration. Also discuss potential ethical considerations
        and implementation challenges.
        """

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        logger.info("Starting optimized generation...")

        # Generate with optimization
        outputs = accelerator.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=1024,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=3
        )

        # Decode output
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print("\nGenerated Response:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)

        # Clear caches
        accelerator.clear_caches()

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")


if __name__ == "__main__":
    main()