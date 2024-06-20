from gradio_client import Client

client = Client("https://6beaa44dc4d4adbff3.gradio.live/")
result = client.predict(
		task="antonyms",
		api_name="/load_task"
)
print(result)