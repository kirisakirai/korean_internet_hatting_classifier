from project_code import model_load, project_test_code
import requests
import json
from twitchio.ext import commands
from sklearn.linear_model import LogisticRegressionCV
import azure.cognitiveservices.speech as speechsdk

# OpenAI API 설정
API_KEY = "open_api_key"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# 모델 불러오기
model, vect_morp = model_load()

def check_hate_speech(text, model, vect_morp):
    return project_test_code(text, model, vect_morp)

# OpenAI GPT-4를 이용한 채팅 완성 생성 함수
def generate_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0.5, max_tokens=200):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print("An error occurred: ", e)

# 텍스트를 음성으로 변환하는 함수
class text_to_speech:
    def __init__(self):
        subscription_key = "azure_subscription_key"
        region = "region"

        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region
        )
        self.speech_config.speech_synthesis_voice_name = 'ko-KR-SoonBokNeural'
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    def speak(self, text, pitch='+15%', rate='+20%'):
        ssml = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='ko-KR'>
            <voice name='ko-KR-SoonBokNeural'>
                <prosody pitch='{pitch}' rate='{rate}'>{text}</prosody>
            </voice>
        </speak>
        """

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config,
                                                         audio_config=self.audio_config)
        speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()

        if speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")

# Twitch 봇 클래스
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token='twitch_token', prefix='!', initial_channels=['channel_name'])

    async def event_ready(self):
        print(f'Ready | {self.nick}')

    async def event_message(self, message):
        print(message.content)
        if message.echo:
            return
        if '!talk' in message.content.lower():
            text = message.content[5:]
            is_hate_speech = check_hate_speech(text, model, vect_morp)
            print(is_hate_speech)
            if is_hate_speech == 1:
                response = "필터링되었습니다."
            else:
                response = generate_chat_completion(text)
            print(response)

            # 텍스트를 음성으로 변환하는 객체 생성
            tts = text_to_speech()
            # 음성으로 변환
            tts.speak(response)

# 봇 실행
bot = Bot()
bot.run()
