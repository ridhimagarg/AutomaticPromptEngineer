import os
import sys
import openai

from demo import get_demo

# openai.api_key = 'sk-proj-A5JCSBhxSuQM0PodFK8mT3BlbkFJpqyNQJn9tY0gaV9RX3cj'
demo = get_demo()
demo.launch(debug=True)