# Local AI Voice Assistant

This is a simple voice assistant. It listens for a wake word (Nebula), and then speaks the response back to you.

## How to Get Started

Getting this up and running is pretty straightforward. You need a few **free** API keys first.

### 1. Get Your API Keys

You'll need API keys from three services. But, you likely won't get to the free plan limits.

- **Picovoice Porcupine:** This is for detecting the "wake word".

  - **Get your key at:** [https://picovoice.ai/platform/porcupine/](https://picovoice.ai/platform/porcupine/)

- **Google Gemini:** This is the LLM.

  - **Get your key at:** [https://aistudio.google.com/](https://aistudio.google.com/)

- **ElevenLabs:** This provides the realistic voice that answers back to you.

  - **Get your key at:** [https://elevenlabs.io/](https://elevenlabs.io/)

### 2. Download the repository

Create a folder and put downloaded files inside it.

### 3. Set Up Your `config.json` File

In the folder assets, you'll find a `config.json` file. Open it with any text editor (like Notepad). You'll need to paste in your API keys.

Here’s what you need to fill out:

```json
{
  "porcupine_key": "PASTE_YOUR_PICOVOICE_KEY_HERE",
  "gemini_key": "PASTE_YOUR_GEMINI_KEY_HERE",
  "elevenlabs_key": "PASTE_YOUR_ELEVENLABS_KEY_HERE",
  "whisper_language": "en"
}
```

You can also change the language, the model supports nearly all languages, use the first two letters of the language.

### 4. Run the Assistant!

Once you've saved your `config.json` just double-click the `.exe` file.

A command window will pop up. When you see the message "Listening for wake word...", you're all set! Say "nebula" and ask a question.
