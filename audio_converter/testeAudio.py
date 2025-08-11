from google import genai

client = genai.Client(api_key="AIzaSyCOzpQcsfQBwutmWZwX5KhYoXxvoKGG85U")   # ou use credenciais configuradas
# upload
myfile = client.files.upload(file="Gravando.m4a")
# usar no prompt
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Descreva esse áudio:", myfile]
)
print(response.text)
