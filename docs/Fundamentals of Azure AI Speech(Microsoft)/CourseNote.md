# Fundamentals of Azure AI Speech
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/recognize-synthesize-speech/)
### Duration: 2 Hours
---

## Course Note

### Speech recognition
- Speech recognition takes the spoken word and converts it into data that can be processed - often by transcribing it into text. 
- Speech patterns are analyzed in the audio to determine recognizable patterns that are mapped to words.
- he software typically uses multiple models, including:
  - An acoustic model that converts the audio signal into phonemes (representations of specific sounds).将音频信号转换为音素（特定声音的表示）的声学模型
  - A language model that maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes. 将音素映射到单词的语言模型，通常使用统计算法根据音素预测最可能的单词序列。

### Speech synthesis 语音合成
- Concern with vocalizing data, usually by converting text to speech.
- A speech synthesis solution typically requires the following information:
  - The text to be spoken
  - The voice to be used to vocalize the speech
- Steps:
  - The system typically tokenizes the text to break it down into individual words
  - Assigns phonetic sounds to each word
  - Breaks the phonetic transcription into prosodic units to create phonemes that will be converted to audio format.
  - These phonemes are then synthesized as audio and can be assigned a particular voice, speaking rate, pitch, and volume.


### The speech to text API in Azure AI
- Real-time transcription 实时转录
  - Real-time speech to text allows you to transcribe text in audio streams. You can use real-time transcription for presentations, demos, or any other scenario where a person is speaking.
  - The application will need to be listening for incoming audio from a microphone, or other audio input source such as an audio file. The application code streams the audio to the service, which returns the transcribed text.
- Batch transcription
  - You can point to audio files with a shared access signature (SAS) URI and asynchronously receive transcription results.
  - run in an asynchronous manner because the batch jobs are scheduled on a best-effort basis. 异步运行
  - Normally a job will start executing within minutes of the request but there is no estimate for when a job changes into the running state. 几分钟后开始，但无法预估具体什么时候开始。

### The text to speech API in Azure AI
- Speech synthesis voices
  - Can specify the voice to be used to vocalize the text.
  - The service includes multiple pre-defined voices with support for multiple languages and regional pronunciation, including neural voices that leverage neural networks to overcome common limitations in speech synthesis with regard to intonation, resulting in a more natural sounding voice. 
  - You can also develop custom voices and use them with the text to speech API