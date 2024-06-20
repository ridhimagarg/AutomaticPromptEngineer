"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Constants for LLaMA 3 costs (example values, you may need to adjust these based on actual costs)
llama_costs_per_thousand = {
    'llama3': 0.0150  # Example cost
}

def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return LLaMA_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "LLaMA_Insert":
        return LLaMA_Insert(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")

class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass

class LLaMA_Forward(LLM):
    """Wrapper for LLaMA 3."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        print("quantized")
        self.tokenizer = AutoTokenizer.from_pretrained(config['gpt_config']['model'])
        self.model = AutoModelForCausalLM.from_pretrained(config['gpt_config']['model'], return_dict=True, load_in_8bit=True, device_map='auto')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += llama_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        # try:
        return fn(prompt, n)
        # except BatchSizeException as e:
        #     if n == 1:
        #         raise e
        #     return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['max_tokens'])
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        config = self.config.copy()
        config['n'] = n
        responses = []
        for p in prompt:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=config['max_tokens'], num_return_sequences=n)
            for i in range(n):
                response = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                responses.append(response)
        return responses

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        return self.__generate_text(prompt, n)

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        config = self.config.copy()
        config['logprobs'] = 1
        config['echo'] = True
        config['max_tokens'] = 0
        responses = []
        for t in text:
            inputs = self.tokenizer(t, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            log_probs = log_probs.tolist()
            responses.append(log_probs)
        return responses, [t for t in text]

class LLaMA_Insert(LLM):

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.tokenizer = AutoTokenizer.from_pretrained(config['gpt_config']['model'])
        self.model = AutoModelForCausalLM.from_pretrained(config['gpt_config']['model'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += llama_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
